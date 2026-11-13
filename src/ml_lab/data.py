"""Download original Fashion-MNIST IDX files and make a stratified split."""

import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


def digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Hash a file incrementally without loading it into memory."""
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root: str | Path) -> dict:
    """Download and verify content against pinned SHA256 digests."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, md5 in FILES.items():
        path = root / name
        if not path.exists():
            temp = path.with_suffix(".part")
            with (
                urllib.request.urlopen(BASE_URL + name, timeout=60) as response,
                temp.open("wb") as f,
            ):
                while block := response.read(1024 * 1024):
                    f.write(block)
            if digest(temp) != SHA256[name]:
                temp.unlink()
                raise ValueError(f"Checksum mismatch: {name}")
            temp.replace(path)
        if digest(path) != SHA256[name]:
            raise ValueError(f"Checksum mismatch: {name}")
        manifest[name] = {"sha256": digest(path), "source": BASE_URL + name}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_idx(path: str | Path) -> torch.Tensor:
    """Read validated IDX labels or flattened, scaled 28 by 28 images."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        raise ValueError("Truncated IDX header")
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2049:
        if len(raw) != 8 + count:
            raise ValueError("Invalid label file length")
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=8).copy()).long()
    if magic == 2051:
        if len(raw) < 16:
            raise ValueError("Truncated image header")
        rows, cols = struct.unpack(">II", raw[8:16])
        if (rows, cols) != (28, 28) or len(raw) != 16 + count * rows * cols:
            raise ValueError("Invalid image shape or length")
        return (
            torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).copy())
            .reshape(count, 784)
            .float()
            / 255
        )
    raise ValueError(f"Unsupported IDX magic: {magic}")


def split_indices(
    labels: torch.Tensor, seed: int = 42, val_fraction: float = 1 / 6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint, stratified indices using an independent generator."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    g = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in labels.unique(sorted=True):
        idx = torch.where(labels == label)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        n = round(len(idx) * val_fraction)
        if not 0 < n < len(idx):
            raise ValueError("Each class needs train and validation examples")
        val.append(idx[:n])
        train.append(idx[n:])
    return torch.cat(train), torch.cat(val)


def training_data(root: str | Path, seed: int) -> tuple:
    """Fit training-only normalization and return train/validation tensors."""
    root = Path(root)
    for name in list(FILES)[:2]:
        if digest(root / name) != SHA256[name]:
            raise ValueError(f"Checksum mismatch: {name}")
    x = read_idx(root / "train-images-idx3-ubyte.gz")
    y = read_idx(root / "train-labels-idx1-ubyte.gz")
    if len(x) != len(y):
        raise ValueError("Image/label count mismatch")
    train, val = split_indices(y, seed)
    # Fit scalar normalization using training pixels only.
    mean, std = x[train].mean().item(), x[train].std().item()
    if std <= 0:
        raise ValueError("Training data has zero variance")
    metadata = {
        "mean": mean,
        "std": std,
        "seed": seed,
        "train_size": len(train),
        "validation_size": len(val),
        "train_indices_sha256": hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(val.numpy().tobytes()).hexdigest(),
    }
    return (x[train] - mean) / std, y[train], (x[val] - mean) / std, y[val], metadata


def test_data(root: str | Path, mean: float, std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Read verified official test files using stored training statistics."""
    root = Path(root)
    for name in list(FILES)[2:]:
        if digest(root / name) != SHA256[name]:
            raise ValueError(f"Checksum mismatch: {name}")
    x = read_idx(root / "t10k-images-idx3-ubyte.gz")
    y = read_idx(root / "t10k-labels-idx1-ubyte.gz")
    if len(x) != len(y):
        raise ValueError("Image/label count mismatch")
    return (x - mean) / std, y
