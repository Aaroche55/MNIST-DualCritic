"""DDPM forward noising, the noise-prediction MSE loss, and ancestral sampling (Ho et al., 2020).

Images x_0 are float32 NHWC in [-1, 1]. With alpha_bar_t = prod_{s<=t} (1 - beta_s)::

    forward:  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,    eps ~ N(0, I)
    loss:     mean((eps - model(x_t, t, label))^2),                          t ~ Uniform{0, ..., T-1}
    reverse:  x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat) + sigma_t * z

Timesteps are 0-indexed: t = 0 is the least noisy step, and sampling runs t = T-1 down to 0.
"""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# model(x_t, t, labels) -> predicted noise, all as described in generator.model
NoiseModel = Callable[[mx.array, mx.array, mx.array], mx.array]


@dataclass(frozen=True)
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # Reverse-process variance: "posterior" uses beta_tilde_t = beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t),
    # "beta" uses beta_t. Ho et al. report similar sample quality for both.
    sampling_variance: str = "posterior"

    def betas(self) -> np.ndarray:
        if self.beta_schedule == "linear":
            return np.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=np.float64)
        raise ValueError(f"Unknown beta_schedule {self.beta_schedule!r}")


class GaussianDiffusion:
    def __init__(self, config: DiffusionConfig = DiffusionConfig()):  # noqa: B008 (frozen dataclass, safe default)
        if config.sampling_variance not in ("posterior", "beta"):
            raise ValueError(f"Unknown sampling_variance {config.sampling_variance!r}")
        self.config = config
        self.num_timesteps = config.num_timesteps

        # Schedules are derived in float64 and stored as float32 arrays indexed by t
        betas = config.betas()
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
        variances = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) if config.sampling_variance == "posterior" else betas
        )

        def as_mx(values: np.ndarray) -> mx.array:
            return mx.array(values.astype(np.float32))

        self.betas = as_mx(betas)
        self.sqrt_alpha_bars = as_mx(np.sqrt(alpha_bars))
        self.sqrt_one_minus_alpha_bars = as_mx(np.sqrt(1.0 - alpha_bars))
        self.recip_sqrt_alphas = as_mx(1.0 / np.sqrt(alphas))
        self.eps_coefs = as_mx(betas / np.sqrt(1.0 - alpha_bars))
        self.sigmas = as_mx(np.sqrt(variances))

    def q_sample(self, x0: mx.array, t: mx.array, noise: mx.array) -> mx.array:
        """Noise clean images x0 (B, H, W, C) to timesteps t (B,)."""
        return (
            self.sqrt_alpha_bars[t][:, None, None, None] * x0
            + self.sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
        )

    def loss(self, model: NoiseModel, x0: mx.array, labels: mx.array, key: mx.array) -> mx.array:
        """Standard DDPM objective: MSE between the true and predicted noise at a uniformly random timestep."""
        t_key, noise_key = mx.random.split(key)
        t = mx.random.randint(0, self.num_timesteps, (x0.shape[0],), key=t_key)
        noise = mx.random.normal(x0.shape, key=noise_key)
        return mx.mean(mx.square(noise - model(self.q_sample(x0, t, noise), t, labels)))

    def p_sample(self, model: NoiseModel, x_t: mx.array, t: int, labels: mx.array, key: mx.array) -> mx.array:
        """One reverse step x_t -> x_{t-1}; no noise is added on the final step (t = 0)."""
        t_batch = mx.full((x_t.shape[0],), t, dtype=mx.int32)
        eps = model(x_t, t_batch, labels)
        mean = self.recip_sqrt_alphas[t] * (x_t - self.eps_coefs[t] * eps)
        if t == 0:
            return mean
        return mean + self.sigmas[t] * mx.random.normal(x_t.shape, key=key)

    def sample(self, model: NoiseModel, labels: mx.array, key: mx.array, image_shape=(28, 28, 1)) -> mx.array:
        """Generate one image per label, returned as (B, H, W, C) in roughly [-1, 1] (not clipped)."""
        init_key, *step_keys = mx.random.split(key, self.num_timesteps + 1)
        x = mx.random.normal((labels.shape[0], *image_shape), key=init_key)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, labels, step_keys[t])
            mx.eval(x)  # evaluate each step so the lazy graph does not grow across all T steps
        return x
