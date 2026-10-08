import argparse
from functools import partial
from pathlib import Path

import numpy as np

from mnist_dualcritic.data import DATASETS, TRANSFORMS, Dataset, SourceDataset

ASCII_RAMP = " .\"-+oOM#@"


def render(images: list[np.ndarray], titles: list[str]) -> str:
    """Render images side by side as ASCII art."""
    gap = "  "
    width = images[0].shape[1]
    lines = [gap.join(title[:width].ljust(width) for title in titles)]
    for row in range(images[0].shape[0]):
        lines.append(gap.join(
            "".join(ASCII_RAMP[int(pixel) * len(ASCII_RAMP) // 256] for pixel in image[row])
            for image in images
        ))
    return "\n".join(lines)


def get_source(args) -> SourceDataset:
    return DATASETS[args.dataset](data_dir=args.data_dir)


def get_transforms(args):
    """Bind the CLI's transform parameters to each selected transform function."""
    params = {
        "rotation": {"max_degrees": args.max_degrees},
        "translation": {"std_pixels": args.std_pixels},
        "brightness": {"min_scale": args.min_scale, "max_scale": args.max_scale},
    }
    names = list(TRANSFORMS) if "all" in args.transform else list(dict.fromkeys(args.transform))
    return [(name, partial(TRANSFORMS[name], **params[name])) for name in names]


def cmd_list(args):
    print("Datasets:")
    for name, cls in DATASETS.items():
        print(f"  {name:<12} {cls.url}")
    print("Transforms:")
    for name in TRANSFORMS:
        print(f"  {name}")


def cmd_download(args):
    get_source(args).download(force=args.force)


def cmd_format(args):
    source = get_source(args)
    source.format()
    print(f"Wrote {source.formatted_images} and {source.formatted_labels}")


def cmd_info(args):
    dataset = Dataset.from_file(args.file) if args.file else get_source(args).prepare()
    print(f"Samples:  {len(dataset)}")
    print(f"Image:    {dataset.images.shape[1]}x{dataset.images.shape[2]} {dataset.images.dtype}")
    print(f"Pixels:   min {dataset.images.min()}, max {dataset.images.max()}, mean {dataset.images.mean():.2f}")
    print("Labels:")
    values, counts = np.unique(dataset.labels, return_counts=True)
    for value, count in zip(values, counts):
        print(f"  {value:>3}: {count:>6} ({count / len(dataset):.1%})")


def cmd_show(args):
    dataset = Dataset.from_file(args.file) if args.file else get_source(args).prepare()
    rng = np.random.default_rng(args.seed)
    end = min(args.index + args.count, len(dataset))
    if args.index >= end:
        raise SystemExit(f"Index {args.index} is out of range for {len(dataset)} samples")

    # Only transform the samples being shown, not the whole dataset
    subset = dataset[args.index:end]
    columns = [("sample" if args.file else "original", subset)]
    columns += [(name, transform(subset, rng)) for name, transform in get_transforms(args)]

    for offset in range(len(subset)):
        print(f"#{args.index + offset}  label={subset.labels[offset]}")
        print(render([d.images[offset] for _, d in columns], [name for name, _ in columns]))
        print()


def cmd_generate(args):
    source = get_source(args)
    output = args.output or args.data_dir / source.name / "augmented.npz"
    transforms = get_transforms(args)
    if args.no_original and not transforms:
        raise SystemExit("Nothing to generate: --no-original was given without any --transform")
    rng = np.random.default_rng(args.seed)

    original = source.prepare()
    dataset = Dataset.empty(*original.images.shape[1:])
    if not args.no_original:
        dataset.extend(original)
    for name, transform in transforms:
        print(f"Applying {name}...")
        dataset.extend(transform(original, rng))

    dataset.save(output)
    print(f"Wrote {len(dataset)} samples to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data", description="Download, inspect and augment image datasets.")
    parser.add_argument("-d", "--dataset", choices=DATASETS, default="mnist", help="source dataset (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="root data directory (default: %(default)s)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transform_options = argparse.ArgumentParser(add_help=False)
    group = transform_options.add_argument_group("transforms")
    group.add_argument("-t", "--transform", action="append", default=[], choices=[*TRANSFORMS, "all"],
                       help="transform to apply; repeat for several, or use 'all'")
    group.add_argument("--seed", type=int, help="random seed for reproducible transforms")
    group.add_argument("--max-degrees", type=float, default=50.0, help="rotation limit in degrees (default: %(default)s)")
    group.add_argument("--std-pixels", type=float, default=2.0, help="translation std dev in pixels (default: %(default)s)")
    group.add_argument("--min-scale", type=float, default=0.5, help="minimum brightness scale (default: %(default)s)")
    group.add_argument("--max-scale", type=float, default=1.5, help="maximum brightness scale (default: %(default)s)")

    sub = subparsers.add_parser("list", help="list available datasets and transforms")
    sub.set_defaults(func=cmd_list)

    sub = subparsers.add_parser("download", help="download the raw dataset (skipped if the checksum matches)")
    sub.add_argument("-f", "--force", action="store_true", help="re-download even if the checksum matches")
    sub.set_defaults(func=cmd_download)

    sub = subparsers.add_parser("format", help="merge the raw files into the formatted dataset")
    sub.set_defaults(func=cmd_format)

    sub = subparsers.add_parser("info", help="print dataset size, image shape and label distribution")
    sub.add_argument("--file", type=Path, help="inspect a generated .npz file instead of the source dataset")
    sub.set_defaults(func=cmd_info)

    sub = subparsers.add_parser("show", parents=[transform_options], help="preview samples as ASCII art")
    sub.add_argument("-i", "--index", type=int, default=0, help="first sample to show (default: %(default)s)")
    sub.add_argument("-n", "--count", type=int, default=1, help="number of samples to show (default: %(default)s)")
    sub.add_argument("--file", type=Path, help="preview a generated .npz file instead of the source dataset")
    sub.set_defaults(func=cmd_show)

    sub = subparsers.add_parser("generate", parents=[transform_options], help="build an augmented dataset and save it as .npz")
    sub.add_argument("-o", "--output", type=Path, help="output path (default: <data-dir>/<dataset>/augmented.npz)")
    sub.add_argument("--no-original", action="store_true", help="only include transformed samples")
    sub.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
