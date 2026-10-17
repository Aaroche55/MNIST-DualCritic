import json

import numpy as np
import pytest

from mnist_dualcritic import data
from mnist_dualcritic.data import Brightness, Dataset, Rotation, Subset, Translation, get_dataset

from .conftest import TEST, TRAIN


@pytest.fixture
def pool(source) -> Subset:
    return get_dataset(source).train


def label_counts(subset: Subset) -> np.ndarray:
    return np.bincount(subset.source.labels()[subset.indices], minlength=10)


class TestPartitions:
    def test_test_is_the_official_test_partition(self, source):
        test = get_dataset(source).test.load()
        np.testing.assert_array_equal(test.images, TEST[0])
        np.testing.assert_array_equal(test.labels, TEST[1])

    def test_train_is_the_official_train_partition(self, pool):
        np.testing.assert_array_equal(pool.load().images, TRAIN[0])
        assert pool.partition == "train"

    def test_unpacks_as_train_then_test(self, source):
        pool, test = get_dataset(source)
        assert (pool.partition, test.partition) == ("train", "test")
        assert not np.intersect1d(pool.indices, test.indices).size


class TestTransform:
    def test_appends_copies_after_originals(self, source, pool):
        dataset = pool.transform(Rotation(), Brightness(seed=1)).load()
        n = len(TRAIN[1])
        assert len(dataset) == 3 * n
        np.testing.assert_array_equal(dataset.images[:n], TRAIN[0])
        np.testing.assert_array_equal(dataset.images[n : 2 * n], source.transformed(Rotation()).images[:n])
        np.testing.assert_array_equal(dataset.labels, np.tile(TRAIN[1], 3))

    def test_chains(self, pool):
        assert pool.transform(Rotation()).transform(Translation()).transforms == (Rotation(), Translation())

    def test_without_original(self, source, pool):
        dataset = pool.transform(Rotation(), include_original=False).load()
        np.testing.assert_array_equal(dataset.images, source.transformed(Rotation()).images[: len(TRAIN[1])])

    def test_nothing_selected_gives_empty_dataset(self, pool):
        subset = pool.transform(include_original=False)
        assert len(subset) == 0
        assert subset.load().images.shape == (0, *TRAIN[0].shape[1:])

    def test_is_lazy(self, source, pool):
        subset = pool.transform(Rotation())
        assert not source.is_cached(Rotation())
        subset.load()
        assert source.is_cached(Rotation())

    def test_len_counts_every_copy(self, pool):
        assert len(pool.transform(Rotation(), Translation())) == 3 * len(TRAIN[1])


class TestSplit:
    def test_sides_partition_the_images(self, pool):
        train, validation = pool.split(10)
        assert len(validation.indices) == 10
        assert not np.intersect1d(train.indices, validation.indices).size
        np.testing.assert_array_equal(np.union1d(train.indices, validation.indices), pool.indices)

    def test_fraction(self, pool):
        _, validation = pool.split(0.25)
        assert len(validation.indices) == len(TRAIN[1]) // 4

    def test_deterministic_per_seed(self, pool):
        np.testing.assert_array_equal(pool.split(10, seed=3)[1].indices, pool.split(10, seed=3)[1].indices)
        assert not np.array_equal(pool.split(10, seed=3)[1].indices, pool.split(10, seed=4)[1].indices)

    @pytest.mark.parametrize("seed", range(5))
    def test_stratified_keeps_class_balance(self, pool, seed):
        fraction = 0.25
        _, validation = pool.split(fraction, seed=seed)
        assert np.all(np.abs(label_counts(validation) - fraction * label_counts(pool)) <= 1)

    def test_augmented_copies_stay_with_their_image(self, source, pool):
        # The leak split-then-transform prevents: a rotated copy of a validation image ending up in train
        train, validation = pool.split(10)
        dataset = train.transform(Rotation()).load()
        rotated = source.transformed(Rotation())
        for i in validation.indices:
            assert not (dataset.images == rotated.images[i]).all(axis=(1, 2)).any()

    def test_sides_are_original_images(self, pool):
        for side in pool.split(10):
            assert side.transforms == ()
            assert side.include_original

    @pytest.mark.parametrize(
        "make", [lambda p: p.transform(Rotation()), lambda p: p.transform(Rotation(), include_original=False)]
    )
    def test_refuses_transformed_subsets(self, pool, make):
        with pytest.raises(ValueError, match="split first, then transform"):
            make(pool).split(10)

    def test_split_of_a_split(self, pool):
        train, validation = pool.split(10)
        smaller, extra = train.split(5)
        assert len(smaller.indices) + len(extra.indices) + len(validation.indices) == len(TRAIN[1])
        assert not np.intersect1d(extra.indices, validation.indices).size

    @pytest.mark.parametrize("size", [0, len(TRAIN[1]), 0.0, 1.0, -3])
    def test_rejects_impossible_sizes(self, pool, size):
        with pytest.raises(ValueError, match=r"validation|hold out"):
            pool.split(size)


class TestKFold:
    def test_validation_sides_partition_the_pool(self, pool):
        folds = pool.kfold(4)
        assert len(folds) == 4
        validations = np.concatenate([validation.indices for _, validation in folds])
        np.testing.assert_array_equal(np.sort(validations), pool.indices)
        assert {len(validation.indices) for _, validation in folds} == {len(TRAIN[1]) // 4}

    def test_train_is_the_complement(self, pool):
        for train, validation in pool.kfold(3):
            np.testing.assert_array_equal(np.union1d(train.indices, validation.indices), pool.indices)
            assert not np.intersect1d(train.indices, validation.indices).size

    def test_folds_are_stratified(self, pool):
        for _, validation in pool.kfold(4):
            # Each fold is the difference of two stratified prefixes, so a class can be off by up to one at each end
            assert np.all(np.abs(label_counts(validation) - label_counts(pool) / 4) < 2)

    def test_refuses_transformed_subsets(self, pool):
        with pytest.raises(ValueError, match="kfold first, then transform"):
            pool.transform(Rotation()).kfold(2)

    @pytest.mark.parametrize("k", [1, len(TRAIN[1]) + 1])
    def test_rejects_invalid_k(self, pool, k):
        with pytest.raises(ValueError, match="k must be"):
            pool.kfold(k)


class TestProvenance:
    def test_describe(self, pool):
        description = pool.transform(Rotation(seed=2)).describe()
        assert description["source"] == "fake"
        assert description["partition"] == "train"
        assert description["images"] == len(TRAIN[1])
        assert description["transforms"] == [
            {"key": Rotation(seed=2).key, "transform": "rotation", "version": 1, "params": Rotation(seed=2).params}
        ]

    def test_indices_hash_identifies_the_selection(self, pool):
        first, second = pool.split(10, seed=0)[1], pool.split(10, seed=1)[1]
        assert first.describe()["indices_sha256"] != second.describe()["indices_sha256"]
        assert first.describe()["indices_sha256"] == pool.split(10, seed=0)[1].describe()["indices_sha256"]

    def test_save_round_trip(self, pool, tmp_path):
        subset = pool.split(10)[0].transform(Rotation())
        saved = subset.save(tmp_path / "train")
        loaded = Dataset.from_directory(tmp_path / "train")
        np.testing.assert_array_equal(loaded.images, saved.images)

        metadata = json.loads((tmp_path / "train" / data.METADATA_FILENAME).read_text())
        assert metadata["indices_sha256"] == subset.describe()["indices_sha256"]
        assert metadata["samples"] == len(subset)
        assert set(metadata["sha256"]) == {data.IMAGES_FILENAME, data.LABELS_FILENAME}

    def test_indices_are_read_only(self, pool):
        with pytest.raises(ValueError, match="read-only"):
            pool.indices[0] = 0
