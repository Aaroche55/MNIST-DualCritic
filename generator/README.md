# Class-conditional MNIST diffusion generator (MLX)

A deliberately small DDPM that generates 28×28 MNIST digits for a requested label. It is written in Apple MLX
and MLX NN only, trained from random initialisation, and sized for an M4 Mac with 16 GB of unified memory.

| File | Contents |
| --- | --- |
| `model.py` | `ConditionalUNet` and `ModelConfig`: the noise-prediction network |
| `diffusion.py` | `GaussianDiffusion` and `DiffusionConfig`: noise schedule, forward noising, loss, sampling |
| `training.py` | Data loading, training loop, validation loss, checkpoint save and load, CLI |
| `sample.py` | Generate a PNG grid of one digit from a checkpoint, CLI |

MLX is a project dependency, so `uv sync` installs it.

## Model

`ConditionalUNet` predicts the noise in a noisy image (default: about 3.7 M parameters; 64 base channels, 256-wide conditioning):

```
x (B, 28, 28, 1) ─ conv ─ ResBlock 64 ─┬─ down ─ ResBlock 128 ─┬─ down ─ 2 × ResBlock 256 (7×7)
                                       │                       └──────── up ─ concat ─ ResBlock 128
                                       └──────────────────────────────── up ─ concat ─ ResBlock 64 ─ conv ─ ε̂ (B, 28, 28, 1)
```

| Input | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `x` | `(B, 28, 28, 1)` | float32 | noisy image x_t (MLX convolutions are channels-last, NHWC) |
| `t` | `(B,)` | int32 | diffusion timestep in `[0, num_timesteps)` |
| `labels` | `(B,)` | int32 | requested digit in `[0, 10)` |
| output | `(B, 28, 28, 1)` | float32 | predicted noise ε̂ |

**Conditioning.** The timestep is encoded with sinusoidal features followed by a two-layer MLP. The digit is a
learned `nn.Embedding(10, emb_dim)` vector. The two are summed into one conditioning vector. Every residual
block projects that vector to its channel count and adds it to its features after the first convolution. So the
requested digit influences every resolution of the network at every denoising step. There is no
classifier, guidance or adversarial term.

Sizes are set in `ModelConfig` (`base_channels`, `channel_mults`, `emb_dim`, `norm_groups`).

## Diffusion objective

The standard DDPM of Ho et al. (2020), configured by `DiffusionConfig`. The defaults are T = 1000 and a linear β
schedule from 1e-4 to 0.02:

- **Forward process:** x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε, with ε ~ N(0, I) and ᾱ_t = ∏ (1 − β_s).
- **Loss:** draw t uniformly from {0, …, T−1} and minimise `mean((ε − model(x_t, t, label))²)`. Nothing else is
  added to the loss.
- **Sampling:** start from x_{T−1} ~ N(0, I) and apply x_{t−1} = (x_t − β_t / √(1 − ᾱ_t) · ε̂) / √α_t + σ_t z
  down to t = 0. No noise is added on the last step. σ_t² is the posterior variance β̃_t by default
  (`sampling_variance="beta"` uses β_t).

To add another β schedule, extend `DiffusionConfig.betas()`. Every setting is saved in each checkpoint's config,
so changing the defaults does not affect checkpoints that already exist.

## Data, normalisation and batch format

The generator builds its data on every run through the `mnist_dualcritic.data` API, in `training.load_data()`:

```python
pool, _ = get_dataset(MNIST())  # official partitions; test is never loaded
train, validation = pool.split(10_000, seed=0)  # stratified, original images only
train = train.transform(Rotation(), Translation(), Brightness())
```

There is no separate preparation step. On first use the data library downloads MNIST and caches each transform's
output under `data/`; later runs read the cache. Loaded datasets have `images` as `(N, 28, 28)` uint8 in [0, 255]
and `labels` as `(N,)` uint8. The generator itself never writes under `data/` and refuses any output path there.

`training.batches()` yields:

- `images`: `(B, 28, 28, 1)` float32, normalised as `x / 127.5 − 1` to [−1, 1];
- `labels`: `(B,)` int32 digits 0–9.

Generated images are mapped back with `round((clip(x, −1, 1) + 1) · 127.5)`.

## How the splits are used

| Split | Samples (defaults) | Used for |
| --- | --- | --- |
| train | 200,000 (50,000 originals plus rotated, translated and brightness-adjusted copies) | Gradient updates. Reshuffled every epoch with a seeded permutation. |
| validation | 10,000 originals | Denoising loss after each epoch, with timesteps and noise fixed by the seed so epochs are comparable. Selects the `best` checkpoint. |
| test | 10,000 originals: the official MNIST test set | **Not used.** |

Validation is a stratified sample held out from the official 60,000-image train partition *before* augmentation,
and only the remaining 50,000 images are augmented, so no copy of a validation image is trained on. Test is the
official partition, unchanged, because the pretrained OCR critic was trained on the official train partition.

**Do not touch the test split until final evaluation.** Do not use it for training, checkpoint selection,
hyperparameter tuning or looking at samples. `training.load_data()` discards it on purpose.

## Training

```sh
uv run python -m generator.training --epochs 20 --run-name ddpm
uv run python -m generator.training --resume outputs/checkpoints/ddpm/last --epochs 30   # continue to epoch 30
```

Other options: `--batch-size` (128), `--lr` (2e-4, Adam), `--seed` (0), `--base-channels` (64), `--timesteps`
(1000) and `--output-dir` (`outputs`). Data options:

- `--validation` (10000): images held out of the train partition, as a count or a fraction such as `0.2`;
- `--split-seed` (0): which images are held out;
- `-t/--transform` (rotation, translation and brightness): augmentation for train, with default parameters;
  repeat it for several. `--no-augment` trains on the originals only;
- `--data-dir` (`data`).

From Python, `DataConfig` also takes transforms with any parameters:

```python
from generator.training import DataConfig, TrainConfig, train
from generator.model import ModelConfig
from generator.diffusion import DiffusionConfig
from mnist_dualcritic.data import Rotation

data = DataConfig(validation=0.2, transforms=(Rotation(max_degrees=30), Rotation(max_degrees=30, seed=1)))
train(TrainConfig(epochs=20, run_name="ddpm"), ModelConfig(), DiffusionConfig(), data)
```

Each epoch's shuffle and training noise come from that epoch's own stream derived from `--seed`. So a run that is
stopped and resumed from `last` sees the same data order as one that ran straight through. Resume with the same
data options the run started with: if the rebuilt train or validation set differs from the one recorded in the
checkpoint, training refuses to continue, because a different split could put validation images into training.

Outputs (`outputs/` is gitignored, so don't commit checkpoints or images):

```
outputs/
    checkpoints/<run>/last/   model.safetensors, optimizer.safetensors, config.json   (written after every epoch)
    checkpoints/<run>/best/   the same, from the epoch with the lowest validation loss
    metrics/<run>.jsonl       {"epoch", "step", "train_loss", "validation_loss", "best_validation_loss"} per epoch
    generated/                PNG grids from sample.py
```

`config.json` records:

- the model and diffusion settings;
- the optimiser settings;
- the training settings, including the seed, the epochs completed and the step;
- the image normalisation;
- the metrics;
- the data settings (`data_config`);
- for the train and validation sets (`data`): source, partition, image count, a hash of the selected image
  indices and the transforms applied, from `Subset.describe()`. Resuming checks this.

## Sampling

```sh
uv run python -m generator.sample --checkpoint outputs/checkpoints/ddpm/best --digit 7 --count 16 --seed 0
# -> outputs/generated/ddpm-best_digit7_n16_seed0.png
```

```python
from generator.training import load_checkpoint
from generator.sample import generate, make_grid, write_png

model, diffusion, config = load_checkpoint("outputs/checkpoints/ddpm/best")
images = generate(model, diffusion, digit=7, count=16, seed=0)  # uint8 (16, 28, 28)
```

The same checkpoint and seed always produce the same images. Sampling runs all T steps, so it is far slower
than one training step.

## Tests

```sh
uv run pytest tests/test_generator_smoke.py
```

The tests use only small random arrays and never read or download MNIST.
