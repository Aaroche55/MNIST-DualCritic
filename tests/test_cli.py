import json

import numpy as np
import pytest

from mnist_dualcritic.cli import main, render
from mnist_dualcritic.data import Brightness, Dataset, Rotation, Translation

from .conftest import TEST, TRAIN

SAMPLES = len(TRAIN[1]) + len(TEST[1])


@pytest.fixture
def run(source, capsys):
    """Run the CLI against the fake dataset and return its stdout."""

    def run(*args: str) -> str:
        command, *rest = args
        main([command, "-d", "fake", "--data-dir", str(source.raw_dir.parents[1]), *rest])
        return capsys.readouterr().out

    return run


def test_render_draws_images_side_by_side():
    black = np.zeros((2, 3), dtype=np.uint8)
    white = np.full((2, 3), 255, dtype=np.uint8)
    assert render([black, white], ["black", "white"]) == "bla  whi\n     @@@\n     @@@"


def test_list_shows_datasets_transforms_and_cache(run):
    output = run("list")
    assert "mnist" in output
    assert "fake" in output
    assert "rotation     seed=0 max_degrees=50.0" in output
    assert "(none)" in output

    run("generate", "-t", "rotation")
    assert Rotation().key in run("list")


def test_generate_caches_and_skips(run, source):
    output = run("generate", "-t", "all", "--seed", "0", "1")
    assert output.count("Generated") == 6
    for transform in (Rotation(seed=1), Translation(seed=0), Brightness(seed=1)):
        assert source.is_cached(transform)

    assert run("generate", "-t", "rotation").startswith("Cached")
    assert run("generate", "-t", "rotation", "--force").startswith("Generated")


def test_generate_passes_parameters(run, source):
    run("generate", "-t", "rotation", "--max-degrees", "10", "--seed", "3")
    assert source.is_cached(Rotation(max_degrees=10, seed=3))


def test_generate_requires_a_transform(run):
    with pytest.raises(SystemExit, match="pass at least one --transform"):
        run("generate")


def test_invalid_parameters_reported(run):
    with pytest.raises(SystemExit, match="Invalid transform parameters"):
        run("generate", "-t", "brightness", "--min-scale", "2")


def test_info_counts_composed_train_partition(run):
    output = run("info", "-t", "rotation", "--seed", "0", "1")
    assert f"Samples:  {3 * len(TRAIN[1])}" in output
    assert "Image:    12x12 uint8" in output

    assert f"Samples:  {len(TRAIN[1])}" in run("info", "-t", "rotation", "--no-original")


def test_info_test_partition(run):
    assert f"Samples:  {len(TEST[1])}" in run("info", "-p", "test")


def test_split_saves_train_validation_and_test(run, source):
    output = run("split", "--validation", "0.25", "-t", "rotation", "--split-seed", "1")
    assert "train" in output
    splits = {name: Dataset.from_directory(source.splits_dir / name) for name in ("train", "validation", "test")}
    held_out = len(TRAIN[1]) // 4
    assert len(splits["validation"]) == held_out
    assert len(splits["train"]) == 2 * (len(TRAIN[1]) - held_out)
    np.testing.assert_array_equal(splits["test"].images, TEST[0])

    metadata = json.loads((source.splits_dir / "train" / "metadata.json").read_text())
    assert metadata["partition"] == "train"
    assert [t["key"] for t in metadata["transforms"]] == [Rotation().key]


def test_split_accepts_a_count_and_output_dir(run, tmp_path):
    run("split", "--validation", "7", "-o", str(tmp_path / "out"))
    assert len(Dataset.from_directory(tmp_path / "out" / "validation")) == 7


@pytest.mark.parametrize("size", ["0", str(len(TRAIN[1])), "1.5"])
def test_split_rejects_impossible_sizes(run, size):
    with pytest.raises(SystemExit, match="Invalid split"):
        run("split", "--validation", size)


def test_info_without_anything_selected(run):
    with pytest.raises(SystemExit, match="Nothing to describe"):
        run("info", "--no-original")


def test_show_previews_original_and_transforms(run):
    output = run("show", "-i", "2", "-n", "2", "-t", "rotation", "-t", "brightness")
    assert f"#2  label={TRAIN[1][2]}" in output
    assert f"#3  label={TRAIN[1][3]}" in output
    assert "original" in output
    assert "rotation" in output
    assert "brightness" in output


def test_show_stops_at_end_of_dataset(run):
    output = run("show", "-i", str(SAMPLES - 1), "-n", "5")
    assert output.count("label=") == 1


@pytest.mark.parametrize("args", [("-i", "-1"), ("-n", "0")])
def test_show_rejects_bad_ranges(run, args):
    with pytest.raises(SystemExit) as exit_info:
        run("show", *args)
    assert exit_info.value.code == 2


def test_show_index_out_of_range(run):
    with pytest.raises(SystemExit, match="out of range"):
        run("show", "-i", str(SAMPLES))


def test_clean_selected_and_all(run, source):
    run("generate", "-t", "rotation", "-t", "brightness")
    assert "Removed" in run("clean", "-t", "rotation")
    assert not source.is_cached(Rotation())
    assert source.is_cached(Brightness())

    run("clean")
    assert source.cached_transforms() == {}
    assert run("clean") == "Nothing to remove\n"
