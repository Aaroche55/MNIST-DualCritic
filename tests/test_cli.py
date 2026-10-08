import numpy as np
import pytest

from mnist_dualcritic.cli import main, render
from mnist_dualcritic.data import Brightness, Rotation, Translation

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


def test_info_counts_composed_dataset(run):
    output = run("info", "-t", "rotation", "--seed", "0", "1")
    assert f"Samples:  {3 * SAMPLES}" in output
    assert "Image:    12x12 uint8" in output

    assert f"Samples:  {SAMPLES}" in run("info", "-t", "rotation", "--no-original")


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
