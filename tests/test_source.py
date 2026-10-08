import json

import numpy as np
import pytest

from mnist_dualcritic import data
from mnist_dualcritic.data import DATASETS, MNIST, Brightness, Dataset, Rotation, get_dataset

from .conftest import RAW_FILES, TEST, TRAIN, FakeDataset


class TestMNISTConfig:
    def test_registered(self):
        assert DATASETS["mnist"] is MNIST

    def test_every_raw_file_has_a_checksum(self):
        raw_paths = {path for pair in MNIST.raw_files for path in pair}
        assert raw_paths == set(MNIST.checksums)

    def test_data_dirs_are_per_dataset(self, tmp_path):
        source = MNIST(tmp_path)
        assert source.raw_dir == tmp_path / "mnist" / "raw"
        assert source.formatted_dir == tmp_path / "mnist" / "formatted"
        assert source.transforms_dir == tmp_path / "mnist" / "transforms"


class TestRawFiles:
    def test_valid(self, raw_source):
        assert raw_source.invalid_raw_files() == []

    def test_unrelated_files_ignored(self, raw_source):
        (raw_source.raw_dir / ".DS_Store").write_bytes(b"junk")
        assert raw_source.invalid_raw_files() == []

    def test_missing_and_corrupt_reported(self, raw_source):
        (raw_source.raw_dir / "train/images.idx3").unlink()
        (raw_source.raw_dir / "test/labels.idx1").write_bytes(b"corrupt")
        assert raw_source.invalid_raw_files() == ["train/images.idx3", "test/labels.idx1"]


class TestDownload:
    def test_skipped_when_files_valid(self, raw_source):
        raw_source.download()  # the autouse no_network fixture fails the test if this downloads

    def test_fetches_when_missing(self, data_dir, serve_zip):
        requests = serve_zip()
        source = FakeDataset(data_dir)
        source.download()
        assert requests == [FakeDataset.url]
        assert source.invalid_raw_files() == []

    def test_force_downloads_even_when_valid(self, raw_source, serve_zip):
        requests = serve_zip()
        raw_source.download(force=True)
        assert len(requests) == 1

    def test_bad_download_raises(self, data_dir, serve_zip):
        serve_zip({**RAW_FILES, "train/images.idx3": b"tampered"})
        with pytest.raises(RuntimeError, match=r"failed checksum verification: train/images\.idx3"):
            FakeDataset(data_dir).download()


class TestFormatAndPrepare:
    def test_format_merges_splits_in_order(self, raw_source):
        raw_source.format()
        dataset = raw_source.load()
        np.testing.assert_array_equal(dataset.images, np.concatenate([TRAIN[0], TEST[0]]))
        np.testing.assert_array_equal(dataset.labels, np.concatenate([TRAIN[1], TEST[1]]))

    def test_prepare_downloads_and_formats_once(self, data_dir, serve_zip):
        requests = serve_zip()
        source = FakeDataset(data_dir)
        assert len(source.prepare()) == len(TRAIN[1]) + len(TEST[1])
        source.prepare()
        assert len(requests) == 1


class TestTransformCache:
    def test_generates_then_loads_from_cache(self, source, monkeypatch):
        transform = Rotation(seed=1)
        assert not source.is_cached(transform)
        generated = source.transformed(transform)
        assert source.is_cached(transform)

        def fail(*args):
            raise AssertionError("should have loaded from cache")

        monkeypatch.setattr(Rotation, "apply", fail)
        cached = source.transformed(transform)
        np.testing.assert_array_equal(cached.images, generated.images)

    def test_cache_matches_fresh_run(self, source):
        transform = Brightness(seed=2)
        source.transformed(transform)
        np.testing.assert_array_equal(source.transformed(transform).images, transform(source.load()).images)

    def test_metadata_describes_output(self, source):
        transform = Rotation(seed=4, max_degrees=20)
        source.transformed(transform)
        metadata = json.loads((source.transforms_dir / transform.key / data.METADATA_FILENAME).read_text())
        assert metadata["key"] == transform.key
        assert metadata["source"] == "fake"
        assert metadata["transform"] == "rotation"
        assert metadata["version"] == Rotation.version
        assert metadata["params"] == {"seed": 4, "max_degrees": 20.0}
        assert metadata["samples"] == len(source.load())
        assert metadata["image_shape"] == list(TRAIN[0].shape[1:])
        assert "created" in metadata

    def test_cached_transforms_lists_outputs(self, source):
        transforms = [Rotation(), Brightness(seed=1)]
        for transform in transforms:
            source.transformed(transform)
        assert sorted(source.cached_transforms()) == sorted(t.key for t in transforms)

    def test_directory_without_metadata_is_not_cached(self, source):
        transform = Rotation()
        source.transformed(transform)
        (source.transforms_dir / transform.key / data.METADATA_FILENAME).unlink()
        assert not source.is_cached(transform)
        assert transform.key not in source.cached_transforms()

    def test_interrupted_regenerate_keeps_previous_output(self, source, monkeypatch):
        transform = Rotation(seed=9)
        previous = source.transformed(transform)

        def crash(self, directory):
            raise KeyboardInterrupt

        monkeypatch.setattr(Dataset, "save", crash)
        with pytest.raises(KeyboardInterrupt):
            source.transformed(transform, regenerate=True)
        monkeypatch.undo()

        assert source.is_cached(transform)
        np.testing.assert_array_equal(source.transformed(transform).images, previous.images)

    def test_partial_directories_ignored_and_replaced(self, source):
        transform = Rotation()
        partial = source.transforms_dir / (transform.key + data.PARTIAL_SUFFIX)
        partial.mkdir(parents=True)
        (partial / data.METADATA_FILENAME).write_text("{}")

        assert source.cached_transforms() == {}
        source.transformed(transform)
        assert not partial.exists()
        assert list(source.cached_transforms()) == [transform.key]


class TestGetDataset:
    def test_composes_original_then_transforms(self, source):
        original = source.load()
        transforms = [Rotation(seed=1), Brightness(seed=2)]
        dataset = get_dataset(source, *transforms)

        n = len(original)
        assert len(dataset) == 3 * n
        np.testing.assert_array_equal(dataset.images[:n], original.images)
        for i, transform in enumerate(transforms, start=1):
            np.testing.assert_array_equal(dataset.images[i * n : (i + 1) * n], source.transformed(transform).images)
        np.testing.assert_array_equal(dataset.labels, np.tile(original.labels, 3))

    def test_without_original(self, source):
        dataset = get_dataset(source, Rotation(), include_original=False)
        np.testing.assert_array_equal(dataset.images, source.transformed(Rotation()).images)

    def test_nothing_selected_gives_empty_dataset(self, source):
        dataset = get_dataset(source, include_original=False)
        assert dataset.images.shape == (0, *TRAIN[0].shape[1:])

    def test_caches_transforms(self, source):
        get_dataset(source, Rotation())
        assert source.is_cached(Rotation())
