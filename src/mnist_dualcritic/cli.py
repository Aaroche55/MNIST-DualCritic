import argparse
import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np

from mnist_dualcritic.data import DATASETS, TRANSFORMS, Dataset, SourceDataset, Transform, get_dataset

ASCII_RAMP = ' ."-+oOM#@'


def render(images: list[np.ndarray], titles: list[str]) -> str:
    """Render images side by side as ASCII art."""
    gap = "  "
    width = images[0].shape[1]
    header = gap.join(title[:width].ljust(width) for title in titles)
    rows = [
        gap.join("".join(ASCII_RAMP[int(pixel) * len(ASCII_RAMP) // 256] for pixel in image[row]) for image in images)
        for row in range(images[0].shape[0])
    ]
    return "\n".join([header, *rows])


def format_params(params: dict) -> str:
    return " ".join(f"{name}={value}" for name, value in params.items())


def transform_params(cls: type[Transform]):
    """A transform's parameters other than the seed, which the CLI handles separately."""
    return [f for f in fields(cls) if f.name != "seed"]


def get_source(args) -> SourceDataset:
    return DATASETS[args.dataset](data_dir=args.data_dir)


def get_transforms(args) -> list[Transform]:
    """One transform per selected name and seed, using the CLI's parameter values."""
    names = list(TRANSFORMS) if "all" in args.transform else list(dict.fromkeys(args.transform))
    try:
        return [
            TRANSFORMS[name](seed=seed, **{f.name: getattr(args, f.name) for f in transform_params(TRANSFORMS[name])})
            for name in names
            for seed in dict.fromkeys(args.seed)
        ]
    except ValueError as error:
        raise SystemExit(f"Invalid transform parameters: {error}") from None


def cmd_list(args):
    print("Datasets:")
    for name, cls in DATASETS.items():
        print(f"  {name:<12} {cls.url}")

    print("Transforms:")
    for name, cls in TRANSFORMS.items():
        print(f"  {name:<12} {format_params({f.name: f.default for f in fields(cls)})}")

    source = get_source(args)
    cached = source.cached_transforms()
    print(f"Cached for {source.name} ({source.transforms_dir}):")
    for key, metadata in cached.items():
        print(f"  {key:<24} {metadata['samples']:>6} samples  {format_params(metadata['params'])}")
    if not cached:
        print("  (none)")


def cmd_download(args):
    get_source(args).download(force=args.force)


def cmd_format(args):
    source = get_source(args)
    source.format()
    print(f"Wrote {source.formatted_dir}")


def cmd_generate(args):
    source = get_source(args)
    transforms = get_transforms(args)
    if not transforms:
        raise SystemExit("Nothing to generate: pass at least one --transform")
    for transform in transforms:
        if source.is_cached(transform) and not args.force:
            print(f"Cached     {transform.key}  {format_params(transform.params)}")
            continue
        source.transformed(transform, regenerate=True)
        print(f"Generated  {transform.key}  {format_params(transform.params)}")


def cmd_clean(args):
    source = get_source(args)
    keys = [t.key for t in get_transforms(args)] if args.transform else list(source.cached_transforms())
    # Also catch directories left behind by interrupted runs, which have no metadata
    if not args.transform and source.transforms_dir.exists():
        keys = sorted({*keys, *(path.name for path in source.transforms_dir.iterdir() if path.is_dir())})
    removed = 0
    for key in keys:
        directory = source.transforms_dir / key
        if directory.is_dir():
            shutil.rmtree(directory)
            print(f"Removed {directory}")
            removed += 1
    if not removed:
        print("Nothing to remove")


def cmd_info(args):
    source = get_source(args)
    transforms = get_transforms(args)
    if args.no_original and not transforms:
        raise SystemExit("Nothing to describe: --no-original was given without any --transform")
    dataset = get_dataset(source, *transforms, include_original=not args.no_original)

    print("Composed from:")
    if not args.no_original:
        print(f"  original   {len(source.load()):>6} samples")
    for transform in transforms:
        print(f"  {transform.key}  {format_params(transform.params)}")
    print(f"Samples:  {len(dataset)}")
    print(f"Image:    {dataset.images.shape[1]}x{dataset.images.shape[2]} {dataset.images.dtype}")
    print(f"Pixels:   min {dataset.images.min()}, max {dataset.images.max()}, mean {dataset.images.mean():.2f}")
    print("Labels:")
    values, counts = np.unique(dataset.labels, return_counts=True)
    for value, count in zip(values, counts, strict=True):
        print(f"  {value:>3}: {count:>6} ({count / len(dataset):.1%})")


def cmd_show(args):
    source = get_source(args)
    original = source.prepare()
    end = min(args.index + args.count, len(original))
    if args.index >= len(original):
        raise SystemExit(f"Index {args.index} is out of range for {len(original)} samples")

    # Show the cached outputs so the preview matches exactly what get_dataset returns
    columns: list[tuple[str, Dataset]] = [("original", original)]
    for transform in get_transforms(args):
        title = transform.name if len(args.seed) == 1 else f"{transform.name} s={transform.seed}"
        columns.append((title, source.transformed(transform)))

    for i in range(args.index, end):
        print(f"#{i}  label={original.labels[i]}")
        print(render([d.images[i] for _, d in columns], [title for title, _ in columns]))
        print()


def bounded_int(minimum: int):
    def parse(value: str) -> int:
        number = int(value)
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {number}")
        return number

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data", description="Download, inspect and augment image datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Given to every subcommand so they can follow it, e.g. `data show -d mnist`
    common_options = argparse.ArgumentParser(add_help=False)
    common_options.add_argument(
        "-d", "--dataset", choices=DATASETS, default="mnist", help="source dataset (default: %(default)s)"
    )
    common_options.add_argument(
        "--data-dir", type=Path, default=Path("./data"), help="root data directory (default: %(default)s)"
    )

    # Shared transform options, generated from each Transform's dataclass fields
    transform_options = argparse.ArgumentParser(add_help=False, parents=[common_options])
    group = transform_options.add_argument_group("transforms")
    group.add_argument(
        "-t",
        "--transform",
        action="append",
        default=[],
        choices=[*TRANSFORMS, "all"],
        help="transform to apply; repeat for several, or use 'all'",
    )
    group.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0],
        help="seed(s) for each transform; several seeds give several augmented copies (default: 0)",
    )
    owners: dict[str, str] = {}
    for name, cls in TRANSFORMS.items():
        for f in transform_params(cls):
            if f.name in owners:
                raise ValueError(f"Transforms {owners[f.name]} and {name} both have a parameter named {f.name!r}")
            owners[f.name] = name
            group.add_argument(
                f"--{f.name.replace('_', '-')}",
                type=f.type,
                default=f.default,
                help=f"{name} parameter (default: %(default)s)",
            )

    sub = subparsers.add_parser(
        "list", parents=[common_options], help="list datasets, transforms and cached transform outputs"
    )
    sub.set_defaults(func=cmd_list)

    sub = subparsers.add_parser(
        "download", parents=[common_options], help="download the raw dataset (skipped if the checksum matches)"
    )
    sub.add_argument("-f", "--force", action="store_true", help="re-download even if the checksum matches")
    sub.set_defaults(func=cmd_download)

    sub = subparsers.add_parser(
        "format", parents=[common_options], help="merge the raw files into the formatted dataset"
    )
    sub.set_defaults(func=cmd_format)

    sub = subparsers.add_parser("generate", parents=[transform_options], help="generate and cache transform outputs")
    sub.add_argument("-f", "--force", action="store_true", help="regenerate even if already cached")
    sub.set_defaults(func=cmd_generate)

    sub = subparsers.add_parser(
        "clean",
        parents=[transform_options],
        help="delete cached transform outputs (all of them unless --transform is given)",
    )
    sub.set_defaults(func=cmd_clean)

    sub = subparsers.add_parser(
        "info", parents=[transform_options], help="print size, image shape and label distribution of a composed dataset"
    )
    sub.add_argument("--no-original", action="store_true", help="exclude the untransformed samples")
    sub.set_defaults(func=cmd_info)

    sub = subparsers.add_parser("show", parents=[transform_options], help="preview samples as ASCII art")
    sub.add_argument(
        "-i", "--index", type=bounded_int(0), default=0, help="first sample to show (default: %(default)s)"
    )
    sub.add_argument(
        "-n", "--count", type=bounded_int(1), default=1, help="number of samples to show (default: %(default)s)"
    )
    sub.set_defaults(func=cmd_show)

    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
