import gzip
import struct

import pytest
import torch

from ml_lab.data import read_idx


def test_idx_parsing_and_normalization(tmp_path):
    path = tmp_path / "images.gz"
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, 2, 28, 28) + bytes([0] * 784 + [255] * 784))
    x = read_idx(path)
    assert x.shape == (2, 784) and x.dtype == torch.float32
    assert x[0].sum() == 0 and x[1].sum() == 784


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        struct.pack(">II", 9999, 3),
        struct.pack(">II", 2049, 3) + b"\x01",
        struct.pack(">IIII", 2051, 1, 27, 28) + bytes(756),
    ],
)
def test_corrupt_idx_is_rejected(tmp_path, payload):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as f:
        f.write(payload)
    with pytest.raises(ValueError):
        read_idx(path)
