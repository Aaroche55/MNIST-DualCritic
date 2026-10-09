"""Smoke tests for the MLX diffusion generator, using small random arrays only (no MNIST is read or downloaded)."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="the generator needs MLX: uv run --with mlx pytest")

from generator.diffusion import DiffusionConfig, GaussianDiffusion  # noqa: E402
from generator.model import ConditionalUNet, ModelConfig  # noqa: E402
from generator.training import (  # noqa: E402
    TrainConfig,
    load_checkpoint,
    load_optimizer_state,
    make_optimizer,
    make_train_step,
    save_checkpoint,
)

BATCH = 4
TINY_MODEL = ModelConfig(base_channels=8, emb_dim=16, norm_groups=4)
TINY_DIFFUSION = DiffusionConfig(num_timesteps=10)


@pytest.fixture
def model():
    mx.random.seed(0)
    model = ConditionalUNet(TINY_MODEL)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    images = mx.array(rng.uniform(-1, 1, size=(BATCH, 28, 28, 1)).astype(np.float32))
    labels = mx.array(rng.integers(0, 10, size=BATCH).astype(np.int32))
    return images, labels


def test_forward_shape(model, batch):
    images, labels = batch
    t = mx.array([0, 3, 7, 9], dtype=mx.int32)
    out = model(images, t, labels)
    assert out.shape == (BATCH, 28, 28, 1)
    assert out.dtype == mx.float32


def test_loss_is_finite_scalar(model, batch):
    loss = GaussianDiffusion(TINY_DIFFUSION).loss(model, *batch, mx.random.key(0))
    assert loss.shape == ()
    assert np.isfinite(loss.item())
    assert loss.item() > 0


def test_train_step_updates_parameters(model, batch):
    diffusion = GaussianDiffusion(TINY_DIFFUSION)
    optimizer = make_optimizer(TrainConfig(learning_rate=1e-3))
    before = np.array(model.conv_out.weight)
    loss = make_train_step(model, diffusion, optimizer)(*batch, mx.random.key(0))
    assert np.isfinite(loss)
    assert not np.array_equal(before, np.array(model.conv_out.weight))


def test_sampling_gives_finite_images(model):
    labels = mx.array([0, 5, 9], dtype=mx.int32)
    samples = GaussianDiffusion(TINY_DIFFUSION).sample(model, labels, mx.random.key(0))
    assert samples.shape == (3, 28, 28, 1)
    assert np.isfinite(np.array(samples)).all()


def test_checkpoint_round_trip(model, batch, tmp_path):
    diffusion = GaussianDiffusion(TINY_DIFFUSION)
    optimizer = make_optimizer(TrainConfig())
    make_train_step(model, diffusion, optimizer)(*batch, mx.random.key(0))
    config = {"model": vars(TINY_MODEL), "diffusion": vars(TINY_DIFFUSION)}
    save_checkpoint(tmp_path / "ckpt", model, optimizer, config)

    loaded, loaded_diffusion, _ = load_checkpoint(tmp_path / "ckpt")
    assert loaded.config == TINY_MODEL
    assert loaded_diffusion.config == TINY_DIFFUSION
    t = mx.array([1, 2, 3, 4], dtype=mx.int32)
    np.testing.assert_array_equal(np.array(model(batch[0], t, batch[1])), np.array(loaded(batch[0], t, batch[1])))

    resumed = make_optimizer(TrainConfig())
    load_optimizer_state(tmp_path / "ckpt", resumed)
    assert resumed.state["step"].item() == optimizer.state["step"].item()
