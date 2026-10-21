"""The MLX port of the pretrained digit classifier and the per-epoch scoring, with random weights (nothing downloaded)."""

import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from generator.classifier import DigitClassifier, classify, read_state_dict, to_mlx_weights
from generator.diffusion import DiffusionConfig, GaussianDiffusion
from generator.model import ConditionalUNet, ModelConfig
from generator.training import (
    OPTIMIZER_FILENAME,
    classifier_records,
    classifier_scores,
    classifier_summary,
    load_checkpoint,
    save_checkpoint,
)

# Shapes of the original PyTorch state dict (conv weights are (out, in, kh, kw); Linear is (out, in))
TORCH_SHAPES = {
    **{
        f"conv_layers.{block * 5}.{name}": shape
        for block, (i, o, k) in enumerate([(1, 32, 3), (32, 64, 3), (64, 128, 3), (128, 256, 1)])
        for name, shape in (("weight", (o, i, k, k)), ("bias", (o,)))
    },
    **{
        f"conv_layers.{block * 5 + 1}.{name}": (o,)
        for block, o in enumerate([32, 64, 128, 256])
        for name in ("weight", "bias", "running_mean", "running_var")
    },
    "fc_layers.1.weight": (128, 256),
    "fc_layers.1.bias": (128,),
    "fc_layers.4.weight": (10, 128),
    "fc_layers.4.bias": (10,),
}


def random_state_dict(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = {name: rng.normal(scale=0.1, size=shape).astype(np.float32) for name, shape in TORCH_SHAPES.items()}
    for name in state:
        if name.endswith("running_var"):
            state[name] = np.abs(state[name]) + 0.5
    return state


@pytest.fixture
def fake_torch(monkeypatch):
    """Just enough of a `torch` module for pickle to write a state dict the way torch.save does."""
    torch, utils = types.ModuleType("torch"), types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args):
        raise AssertionError("only used as a pickle reference")

    FloatStorage.__module__, _rebuild_tensor_v2.__module__ = "torch", "torch._utils"
    FloatStorage.__qualname__, _rebuild_tensor_v2.__qualname__ = "FloatStorage", "_rebuild_tensor_v2"
    torch.FloatStorage, utils._rebuild_tensor_v2 = FloatStorage, _rebuild_tensor_v2
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._utils", utils)
    return torch


def torch_zip(state: dict[str, np.ndarray], torch) -> bytes:
    """A torch.save-style zip: each tensor's raw bytes in data/<key>, and a pickle that rebuilds them."""

    class Storage:
        def __init__(self, key, array):
            self.key, self.array = key, array

    class Tensor:
        def __init__(self, storage, shape):
            self.storage, self.shape = storage, shape

        def __reduce__(self):
            stride = tuple(int(np.prod(self.shape[i + 1 :])) for i in range(len(self.shape)))
            return sys.modules["torch._utils"]._rebuild_tensor_v2, (
                self.storage,
                0,
                self.shape,
                stride,
                False,
                OrderedDict(),
            )

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", torch.FloatStorage, obj.key, "cpu", obj.array.size)
            return None

    tensors = {name: Tensor(Storage(str(i), array), array.shape) for i, (name, array) in enumerate(state.items())}
    payload = io.BytesIO()
    Pickler(payload, protocol=2).dump(OrderedDict(tensors))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model/data.pkl", payload.getvalue())
        archive.writestr("model/byteorder", "little")
        for tensor in tensors.values():
            archive.writestr(f"model/data/{tensor.storage.key}", tensor.storage.array.astype("<f4").tobytes())
    return buffer.getvalue()


def test_reads_a_torch_state_dict_without_torch(fake_torch, tmp_path):
    state = random_state_dict()
    path = tmp_path / "model.pt"
    path.write_bytes(torch_zip(state, fake_torch))
    loaded = read_state_dict(path)
    assert loaded.keys() == state.keys()
    for name, array in state.items():
        np.testing.assert_array_equal(loaded[name], array)


def test_refuses_other_globals(tmp_path):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", pickle.dumps({"x": print}, protocol=2))
        archive.writestr("model/byteorder", "little")
    with pytest.raises(pickle.UnpicklingError, match=r"Refusing to load .*print"):
        read_state_dict(path)


def test_weights_map_onto_every_parameter():
    classifier = DigitClassifier()
    weights = to_mlx_weights(random_state_dict())
    assert {name for name, _ in weights} == {name for name, _ in tree_flatten(classifier.parameters())}
    classifier.load_weights(weights)  # strict: shapes and names must match
    classifier.eval()
    probabilities = classify(classifier, np.zeros((3, 28, 28), dtype=np.uint8))
    assert probabilities.shape == (3, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_records_and_summary():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.full((4, 10), 0.01, dtype=np.float32)
    probabilities[[0, 1, 2, 3], [0, 3, 1, 1]] = 0.91
    records = classifier_records(labels, probabilities)
    assert [r["sample"] for r in records] == [0, 1, 0, 1]
    assert [r["predicted"] for r in records] == [0, 3, 1, 1]
    assert records[1] == records[1] | {"correct": False, "target_probability": pytest.approx(0.01)}
    summary = classifier_summary(labels, probabilities)
    assert summary["classifier_accuracy"] == 0.75
    assert summary["classifier_confidence"] == pytest.approx(0.91)
    assert summary["classifier_accuracy_per_digit"][:2] == [0.5, 1.0]


def test_scores_are_repeatable():
    mx.random.seed(0)
    model = ConditionalUNet(ModelConfig(base_channels=8, emb_dim=16, norm_groups=4))
    classifier = DigitClassifier()
    classifier.load_weights(to_mlx_weights(random_state_dict()))
    classifier.eval()
    diffusion = GaussianDiffusion(DiffusionConfig(num_timesteps=5))
    images, labels, probabilities = classifier_scores(model, diffusion, classifier, per_class=2, seed=0)
    assert images.shape == (20, 28, 28)
    assert images.dtype == np.uint8
    assert labels.tolist() == [d for d in range(10) for _ in range(2)]
    assert probabilities.shape == (20, 10)
    np.testing.assert_array_equal(images, classifier_scores(model, diffusion, classifier, 2, seed=0)[0])


def test_weights_only_checkpoint_can_be_sampled(tmp_path):
    config = ModelConfig(base_channels=8, emb_dim=16, norm_groups=4)
    model = ConditionalUNet(config)
    save_checkpoint(tmp_path / "epoch-001", model, None, {"model": vars(config), "diffusion": {"num_timesteps": 5}})
    assert not (tmp_path / "epoch-001" / OPTIMIZER_FILENAME).exists()
    loaded, _, _ = load_checkpoint(tmp_path / "epoch-001")
    assert loaded.config == config
