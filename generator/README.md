# Class-conditional MNIST diffusion generator (MLX)

A deliberately small DDPM that generates 28×28 MNIST digits for a requested label. It is written in Apple MLX
and MLX NN only, trained from random initialisation, and sized for an M4 Mac with 16 GB of unified memory.

| File | Contents |
| --- | --- |
| `model.py` | `ConditionalUNet` and `ModelConfig`: the noise-prediction network |
| `diffusion.py` | `GaussianDiffusion` and `DiffusionConfig`: noise schedule, forward noising, loss, sampling |
| `training.py` | Data loading, training loop, validation loss, checkpoint save and load, CLI |
| `sample.py` | Generate a PNG grid of one digit from a checkpoint, CLI |
| `classifier.py` | `DigitClassifier`: the pretrained MNIST classifier that scores generated samples, ported to MLX |
| `score.py` | Score saved checkpoints with the classifier, CLI |
| `images.py` | `make_grid` and `write_png` |
| `plot.py` | Plot logged metrics to PNG, PDF or SVG, CLI |

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
(1000) and `--output-dir` (`outputs`). Saving and evaluation options:

- `--save-every-epoch`: also keep each epoch's weights in `checkpoints/<run>/epoch-NNN/`, so any epoch can be
  sampled later. These hold the model and `config.json` but not the optimiser state (about 15 MB each with the
  default model), so resume from `last`, not from them;
- `--classifier-samples` (16): images of each digit generated and scored by the pretrained classifier after
  every epoch (see [Classifier evaluation](#classifier-evaluation)); `0` turns it off.

Data options:

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
    checkpoints/<run>/epoch-NNN/   model.safetensors, config.json   (with --save-every-epoch)
    metrics/<run>.jsonl       {"epoch", "step", "train_loss", "validation_loss", "best_validation_loss",
                               "classifier_accuracy", ...} per epoch
    metrics/<run>_classifier.jsonl   one row per generated sample per epoch (see below)
    generated/<run>/epoch-NNN.png    the samples scored that epoch: one row per digit, 0 at the top
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

## Classifier evaluation

There is no discriminator. Sample quality is measured with an off-the-shelf MNIST classifier,
[abdurafay19/Digit-Classifier](https://huggingface.co/abdurafay19/Digit-Classifier) (MIT; a 160 k-parameter CNN,
99.43% on the MNIST test set). `classifier.py` re-implements its architecture in MLX and reads the PyTorch
`model.pt` without PyTorch, through an unpickler that accepts only the names a plain state dict needs. The weights
are pinned to one commit, checked against their SHA-256 and downloaded once into `.cache/digit-classifier/`
(gitignored). The port's probabilities match the PyTorch model to within 1e-6.

After every epoch, training generates `--classifier-samples` images of each digit from the same starting noise
(fixed by `--seed`), so differences between epochs come from the model. It then classifies them. Each sample gives
one row in `metrics/<run>_classifier.jsonl`:

```json
{"epoch": 3, "step": 4689, "target": 7, "sample": 0, "predicted": 7, "correct": true,
 "confidence": 0.91, "target_probability": 0.91, "probabilities": [0.009, ..., 0.91, 0.01, 0.01]}
```

`target` is the requested digit, `sample` its index (0 to N−1), `predicted` the classifier's guess, `confidence`
its probability and `target_probability` the probability it gives the requested digit. The epoch's summary goes into
`metrics/<run>.jsonl` and the checkpoint's `config.json`: `classifier_accuracy`, `classifier_confidence` and
`classifier_target_probability` (means over all samples), and `classifier_accuracy_per_digit` (ten values).
With the default 16 per digit, evaluation adds about a minute per epoch on an M-series Mac, since sampling runs
all T steps.

The classifier was trained with label smoothing, so it is not confident even on real digits. On 2,000 real
validation images it is 99% accurate with a mean confidence of 0.91. Compare against those figures, not 1.0.

For plots, load the files with pandas:

```python
import pandas as pd

epochs = pd.read_json("outputs/metrics/ddpm.jsonl", lines=True)
samples = pd.read_json("outputs/metrics/ddpm_classifier.jsonl", lines=True)
samples.groupby(["epoch", "target"]).correct.mean().unstack().plot()  # accuracy per digit over training
pd.crosstab(samples.target, samples.predicted)  # confusion matrix
```

To score checkpoints that already exist, for example every epoch of a run saved with `--save-every-epoch`, or
with more samples:

```sh
uv run python -m generator.score --checkpoint outputs/checkpoints/ddpm/epoch-* --samples 16
# -> rows tagged with "checkpoint" and "epoch" appended to outputs/metrics/scores.jsonl
```

## Plots

`plot.py` draws figures from the metrics files and writes them to `outputs/plots/` (or `-o`):

```sh
uv run python -m generator.plot --run ddpm                       # all panels -> outputs/plots/ddpm.png
uv run python -m generator.plot --run ddpm -p loss -p confidence -o figures/ddpm.pdf
uv run python -m generator.plot --run ddpm --separate --format svg   # one file per panel, for slides
uv run python -m generator.plot --run ddpm --run ddpm-small -p accuracy   # compare runs on one panel
uv run python -m generator.plot --run ddpm16+ddpm16-classifier   # a run joined with its resumed continuation
uv run python -m generator.plot --run ddpm --list                # what can be plotted
uv run python -m generator.plot --run ddpm -p digit-bars --separate   # a bar chart for every epoch
uv run python -m generator.plot --run ddpm -p digit-bars --epoch 1 10 20 --columns 3   # chosen epochs side by side
uv run python -m generator.plot --run ddpm -p digit-bars-stacked --epoch 5 15 25          # those epochs overlapped
uv run python -m generator.plot --run ddpm -p digit-bars-stacked --bar-value confidence   # mean confidence instead
```

| Panel (`-p`) | Shows |
| --- | --- |
| `loss` | train and validation denoising loss per epoch |
| `accuracy` | fraction of generated samples the classifier labels as the requested digit |
| `confidence` | mean classifier confidence and probability of the requested digit, against real digits (0.91) |
| `digit-accuracy` | accuracy for each digit, one line per digit |
| `digit-confidence` | mean confidence for each digit |
| `digit-accuracy-map`, `digit-confidence-map` | the same as a heatmap, digits down and epochs across; easier to read than ten lines once a run has many epochs |
| `confusion` | requested digit against the classifier's guess, for each `--epoch` (default: the latest) |
| `digit-bars` | bar chart per digit, one per `--epoch` (default: every scored epoch), with counts such as 14/16 and the previous scored epoch outlined |
| `digit-bars-stacked` | the `--epoch` epochs (default: every scored epoch) overlapped on one bar chart, shaded dark (early) to light (late) |

The bar charts show accuracy by default. `--bar-value confidence` shows the classifier's mean confidence in its
guess, and `--bar-value target-probability` shows the mean probability it gives the requested digit. In
`digit-bars-stacked`, each digit's bars are drawn tallest at the back and shortest at the front, so every epoch stays
visible. Each bar further forward is slightly narrower, so an epoch that ties a later one (common with 16 samples
per digit) still shows as a coloured rim. Colour follows time: a ramp from dark purple for the earliest chosen epoch
to yellow-green for the latest, evenly spaced by order.

Any numeric key in `<run>.jsonl`, such as `best_validation_loss`, also works as a panel. With no `-p`, every panel
except `digit-confidence`, the heatmaps and the bar charts is drawn. Each line has its own colour (validation and the requested-digit probability are also dashed, for greyscale print), and past 30 epochs the per-epoch dots are dropped so long runs stay readable. Comparing several runs puts the per-epoch panels on shared axes and draws
the per-digit panels once per run. Other options: `--columns` (2), `--size W H` (inches per panel), `--dpi` (200)
and `--title`. The file type comes from `--format` or the extension of `-o`. PDF and SVG are vector formats and
stay sharp in papers and slides.

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
uv run pytest tests/test_generator_smoke.py tests/test_classifier.py
```

The tests use only small random arrays and never read or download MNIST or the classifier weights.
