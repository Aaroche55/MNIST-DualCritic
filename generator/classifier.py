"""Pretrained MNIST digit classifier, ported to MLX, used to score the generator's samples.

The weights are abdurafay19/Digit-Classifier from the Hugging Face Hub (MIT licence; 99.43% on the MNIST test set),
trained in PyTorch on the official MNIST train partition. The architecture is re-implemented here in MLX and the
PyTorch state dict is read without PyTorch: ``model.pt`` is a zip of raw little-endian tensor storages plus a pickle
that says how to assemble them, and the pickle is read with an unpickler that refuses anything except the three
names a plain state dict needs.

Input: uint8 greyscale images (B, 28, 28), white digit on black, as MNIST and the generator produce. They are
scaled to [0, 1] then normalised with mean 0.1307 and std 0.3081, as in the classifier's training.
Output: softmax probabilities (B, 10) as float32 numpy.
"""

import hashlib
import pickle
import zipfile
from pathlib import Path
from urllib.request import urlopen

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO = "abdurafay19/Digit-Classifier"
REVISION = "4167864b6d9f613cca0b92e8f71399f55304cd29"  # pinned commit, so the weights cannot change underneath us
WEIGHTS_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.pt"
WEIGHTS_SHA256 = "6818a763ec8dc86da096699f279d59c5459c142c373a1f5b57413d1a47dcc907"
CACHE_DIR = Path(".cache") / "digit-classifier"

MEAN, STD = 0.1307, 0.3081


class DigitClassifier(nn.Module):
    """Four conv blocks (conv, batch norm, ReLU, 2x2 max pool) then a two-layer MLP, as in the original model.py.

    Spatial sizes: 28 -> 14 -> 7 -> 3 -> 1, with 32, 64, 128 and 256 channels. Dropout is omitted: it is the
    identity at inference, and this model is never trained here.
    """

    def __init__(self):
        super().__init__()
        widths = [(1, 32, 3), (32, 64, 3), (64, 128, 3), (128, 256, 1)]
        self.convs = [nn.Conv2d(i, o, kernel_size=k, padding=k // 2) for i, o, k in widths]
        self.norms = [nn.BatchNorm(o) for _, o, _ in widths]
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x: mx.array) -> mx.array:
        """x (B, 28, 28, 1) normalised float32 -> logits (B, 10)."""
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = self.pool(nn.relu(norm(conv(x))))
        return self.fc2(nn.relu(self.fc1(x.reshape(x.shape[0], -1))))


STORAGE_DTYPES = {"FloatStorage": np.float32, "LongStorage": np.int64}


class _StateDict(dict):
    """Stands in for OrderedDict; torch.save also attaches a ``_metadata`` attribute, which is kept but unused."""


class _StateDictUnpickler(pickle.Unpickler):
    """Reads a PyTorch state dict pickle into numpy arrays; any other global is refused, so nothing else can run."""

    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module: str, name: str):
        if (module, name) == ("collections", "OrderedDict"):
            return _StateDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in STORAGE_DTYPES:
            return STORAGE_DTYPES[name]
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from the classifier weights")

    def persistent_load(self, pid):
        kind, dtype, key, _location, numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"Unexpected persistent id {pid!r}")
        storage = np.frombuffer(self.archive.read(f"{self.prefix}/data/{key}"), dtype=np.dtype(dtype).newbyteorder("<"))
        if storage.size != numel:
            raise pickle.UnpicklingError(f"Storage {key} has {storage.size} elements, expected {numel}")
        return storage


def _rebuild_tensor(storage, offset, size, stride, *_):
    itemsize = storage.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    )
    return np.array(view)  # copy, so the result owns contiguous memory


def read_state_dict(path: Path) -> dict[str, np.ndarray]:
    """{name: array} from a PyTorch zip checkpoint (torch.save of a state dict), without importing torch."""
    with zipfile.ZipFile(path) as archive:
        (pickle_name,) = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        prefix = pickle_name.removesuffix("/data.pkl")
        if archive.read(f"{prefix}/byteorder").decode() != "little":
            raise ValueError(f"{path} is not little-endian")
        with archive.open(pickle_name) as file:
            return dict(_StateDictUnpickler(file, archive, prefix).load())


def to_mlx_weights(state_dict: dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
    """Map PyTorch names and layouts onto DigitClassifier: conv weights go from (O, I, H, W) to (O, H, W, I)."""
    weights = []
    for block in range(4):
        conv, norm = f"conv_layers.{block * 5}", f"conv_layers.{block * 5 + 1}"
        weights += [
            (f"convs.{block}.weight", state_dict[f"{conv}.weight"].transpose(0, 2, 3, 1)),
            (f"convs.{block}.bias", state_dict[f"{conv}.bias"]),
            (f"norms.{block}.weight", state_dict[f"{norm}.weight"]),
            (f"norms.{block}.bias", state_dict[f"{norm}.bias"]),
            (f"norms.{block}.running_mean", state_dict[f"{norm}.running_mean"]),
            (f"norms.{block}.running_var", state_dict[f"{norm}.running_var"]),
        ]
    for ours, theirs in (("fc1", "fc_layers.1"), ("fc2", "fc_layers.4")):
        weights += [(f"{ours}.weight", state_dict[f"{theirs}.weight"]), (f"{ours}.bias", state_dict[f"{theirs}.bias"])]
    return [(name, mx.array(np.ascontiguousarray(value, dtype=np.float32))) for name, value in weights]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_weights(cache_dir: Path = CACHE_DIR) -> Path:
    """The pinned model.pt, downloaded once into `cache_dir` and checked against its SHA-256."""
    path = Path(cache_dir) / REVISION / "model.pt"
    if path.exists() and _sha256(path) == WEIGHTS_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with urlopen(WEIGHTS_URL, timeout=60) as response:
        partial.write_bytes(response.read())
    if (digest := _sha256(partial)) != WEIGHTS_SHA256:
        partial.unlink()
        raise ValueError(f"Downloaded classifier weights have SHA-256 {digest}, expected {WEIGHTS_SHA256}")
    partial.rename(path)
    return path


def load_classifier(weights: Path | None = None) -> DigitClassifier:
    """The pretrained classifier in inference mode; downloads the pinned weights on first use."""
    classifier = DigitClassifier()
    classifier.load_weights(to_mlx_weights(read_state_dict(weights or download_weights())))
    classifier.eval()  # batch norm uses its running statistics
    mx.eval(classifier.parameters())
    return classifier


def classify(classifier: DigitClassifier, images: np.ndarray) -> np.ndarray:
    """uint8 (B, 28, 28) white-on-black digits -> softmax probabilities (B, 10) float32."""
    x = mx.array((images.astype(np.float32) / 255.0 - MEAN) / STD)[..., None]
    return np.array(mx.softmax(classifier(x), axis=-1))
