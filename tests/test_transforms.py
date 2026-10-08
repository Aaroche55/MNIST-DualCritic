from dataclasses import dataclass

import numpy as np
import pytest

from mnist_dualcritic.data import TRANSFORMS, Brightness, Dataset, Rotation, Transform, Translation

from .conftest import TRAIN

ALL_TRANSFORMS = [cls() for cls in TRANSFORMS.values()]


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(*TRAIN)


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=lambda t: t.name)
def test_preserves_shape_dtype_and_labels(dataset, transform):
    output = transform(dataset)
    assert output.images.shape == dataset.images.shape
    assert output.images.dtype == np.uint8
    np.testing.assert_array_equal(output.labels, dataset.labels)


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=lambda t: t.name)
def test_does_not_modify_input(dataset, transform):
    before = dataset.images.copy()
    transform(dataset)
    np.testing.assert_array_equal(dataset.images, before)


@pytest.mark.parametrize("cls", TRANSFORMS.values(), ids=lambda cls: cls.name)
def test_deterministic_per_seed(dataset, cls):
    np.testing.assert_array_equal(cls(seed=3)(dataset).images, cls(seed=3)(dataset).images)
    assert not np.array_equal(cls(seed=3)(dataset).images, cls(seed=4)(dataset).images)


@pytest.mark.parametrize(
    "transform",
    [Rotation(max_degrees=0), Translation(std_pixels=0), Brightness(min_scale=1, max_scale=1)],
    ids=lambda t: t.name,
)
def test_zero_strength_is_identity(dataset, transform):
    np.testing.assert_array_equal(transform(dataset).images, dataset.images)


def test_rotation_stays_within_limit(dataset):
    # A 90 degree rotation of the blob would land outside the columns it starts in; 5 degrees can't
    output = Rotation(max_degrees=5)(dataset)
    assert not output.images[:, :, :2].any()
    assert not output.images[:, :, -2:].any()


def test_translation_moves_whole_pixels(dataset):
    output = Translation(std_pixels=1)(dataset)
    # Integer shifts of a blob away from the border move pixels without resampling them
    for before, after in zip(dataset.images, output.images, strict=True):
        assert sorted(before[before > 0]) == sorted(after[after > 0])


def test_brightness_scales_within_range(dataset):
    output = Brightness(min_scale=0.5, max_scale=0.5)(dataset)
    np.testing.assert_array_equal(output.images, np.rint(dataset.images * 0.5).astype(np.uint8))


def test_brightness_clips_instead_of_wrapping(dataset):
    output = Brightness(min_scale=4, max_scale=4)(dataset)
    assert output.images.max() == 255
    np.testing.assert_array_equal(output.images == 255, dataset.images >= 64)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Rotation(max_degrees=-1), "max_degrees"),
        (lambda: Rotation(max_degrees=181), "max_degrees"),
        (lambda: Translation(std_pixels=-0.1), "std_pixels"),
        (lambda: Brightness(min_scale=2, max_scale=1), "min_scale"),
        (lambda: Brightness(min_scale=-1, max_scale=1), "min_scale"),
    ],
)
def test_rejects_invalid_parameters(make, message):
    with pytest.raises(ValueError, match=message):
        make()


class TestKey:
    def test_int_and_float_parameters_share_a_key(self):
        assert Rotation(max_degrees=50).key == Rotation(max_degrees=50.0).key
        assert isinstance(Rotation(max_degrees=50).max_degrees, float)

    def test_changes_with_parameters_and_seed(self):
        keys = {Rotation().key, Rotation(seed=1).key, Rotation(max_degrees=10).key, Translation().key}
        assert len(keys) == 4

    def test_is_prefixed_with_name(self):
        assert Brightness().key.startswith("brightness-")

    def test_changes_with_version(self):
        @dataclass(frozen=True)
        class RotationV2(Rotation):
            version = 2

        assert RotationV2().key != Rotation().key

    def test_params_include_seed(self):
        assert Translation(seed=5).params == {"seed": 5, "std_pixels": 2.0}


class TestRegistry:
    def test_builtin_transforms_registered(self):
        assert TRANSFORMS["rotation"] is Rotation
        assert TRANSFORMS["translation"] is Translation
        assert TRANSFORMS["brightness"] is Brightness

    def test_duplicate_name_rejected(self):
        with pytest.raises(ValueError, match="already registered"):

            class Duplicate(Transform):
                name = "rotation"

    def test_subclass_without_own_name_not_registered(self):
        before = dict(TRANSFORMS)

        class Unnamed(Rotation):
            pass

        assert before == TRANSFORMS
