import numpy as np
import pytest

from mnist_dualcritic.data import Dataset

from .conftest import TEST, TRAIN


def test_converts_inputs_to_arrays():
    dataset = Dataset(images=[[[1, 2], [3, 4]]], labels=[7])
    assert isinstance(dataset.images, np.ndarray)
    assert isinstance(dataset.labels, np.ndarray)


@pytest.mark.parametrize(
    ("images", "labels", "message"),
    [
        (np.zeros((2, 4)), np.zeros(2), "images must have shape"),
        (np.zeros((2, 4, 4)), np.zeros((2, 1)), "labels must have shape"),
        (np.zeros((2, 4, 4)), np.zeros(3), "sizes do not match"),
    ],
)
def test_validates_shapes(images, labels, message):
    with pytest.raises(ValueError, match=message):
        Dataset(images=images, labels=labels)


def test_empty():
    dataset = Dataset.empty(5, 6)
    assert dataset.images.shape == (0, 5, 6)
    assert dataset.images.dtype == np.uint8
    assert len(dataset) == 0


def test_indexing_keeps_leading_axis():
    dataset = Dataset(*TRAIN)
    single = dataset[2]
    assert single.images.shape == (1, *TRAIN[0].shape[1:])
    np.testing.assert_array_equal(single.labels, TRAIN[1][[2]])

    sliced = dataset[1:4]
    assert len(sliced) == 3
    np.testing.assert_array_equal(sliced.images, TRAIN[0][1:4])


def test_concatenate_and_extend_agree():
    joined = Dataset.concatenate([Dataset(*TRAIN), Dataset(*TEST)])
    extended = Dataset(*TRAIN)
    extended.extend(Dataset(*TEST))

    assert len(joined) == len(TRAIN[1]) + len(TEST[1])
    np.testing.assert_array_equal(joined.images, extended.images)
    np.testing.assert_array_equal(joined.labels, np.concatenate([TRAIN[1], TEST[1]]))


def test_save_round_trip(tmp_path):
    original = Dataset(*TRAIN)
    original.save(tmp_path / "saved")
    loaded = Dataset.from_directory(tmp_path / "saved")
    np.testing.assert_array_equal(loaded.images, original.images)
    np.testing.assert_array_equal(loaded.labels, original.labels)
