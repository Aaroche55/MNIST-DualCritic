"""The generator's data: built through the mnist_dualcritic API, using the small fake dataset (nothing downloaded)."""

import json

import numpy as np
import pytest

from generator.training import DataConfig, data_record, ensure_same_data, load_data
from mnist_dualcritic.data import Rotation, get_dataset

from .conftest import TRAIN


@pytest.fixture
def config(source, data_dir) -> DataConfig:
    return DataConfig(dataset="fake", data_dir=str(data_dir), validation=10, transforms=(Rotation(),))


def test_validation_is_held_out_before_augmenting(config):
    train, validation = load_data(config)
    assert len(validation) == 10
    assert validation.transforms == ()
    assert train.transforms == (Rotation(),)
    assert len(train) == 2 * (len(TRAIN[1]) - 10)
    assert not np.intersect1d(train.indices, validation.indices).size


def test_never_touches_the_test_partition(config, source):
    train, validation = load_data(config)
    test = get_dataset(source).test
    assert not np.intersect1d(np.union1d(train.indices, validation.indices), test.indices).size


def test_split_follows_the_config(config):
    first = load_data(config)[1].indices
    assert np.array_equal(first, load_data(config)[1].indices)
    assert not np.array_equal(first, load_data(DataConfig(**{**vars(config), "split_seed": 1}))[1].indices)


def test_record_is_json_ready(config):
    record = DataConfig(transforms=(Rotation(seed=2),)).record()
    assert record["transforms"] == [{"name": "rotation", "params": {"seed": 2, "max_degrees": 50.0}}]
    json.dumps(config.record())


def test_resume_accepts_the_same_data(config):
    saved = {"data": json.loads(json.dumps(data_record(*load_data(config))))}
    ensure_same_data(saved, *load_data(config))


@pytest.mark.parametrize("change", [{"split_seed": 1}, {"validation": 0.5}, {"transforms": ()}])
def test_resume_refuses_different_data(config, change):
    saved = {"data": data_record(*load_data(config))}
    with pytest.raises(ValueError, match="trained on different data"):
        ensure_same_data(saved, *load_data(DataConfig(**{**vars(config), **change})))
