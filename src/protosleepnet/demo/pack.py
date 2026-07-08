"""Binary packing helpers for the demo static bundle.

The frontend reads these as raw typed arrays (``fetch`` -> ``ArrayBuffer`` ->
``Float32Array`` / ``Uint8Array`` / ``Int16Array``), so the on-disk byte
layout must match exactly. All arrays are little-endian (x86/ARM default,
and what JS ``DataView``/typed arrays assume on all target platforms).

Conventions
-----------
- ``xy``            float32  (n, 2)      UMAP coordinates
- ``label``/``pred``/``proto``  uint8 (n,)   class / prototype indices
- ``proba``        uint8    (n, 5)      softmax * 255 (rounded); JS divides by 255
- ``dist``         float32  (n,)        L2 distance to the matched prototype
- ``subj``/``epoch`` uint16 (n,)        subject index / within-subject epoch index
- raw signal       float16  (n, C, S)   filtered 100 Hz waveform (physioex ``raw`` pipeline)
"""
import numpy as np


def _le(arr: np.ndarray, dtype) -> np.ndarray:
    """Cast to ``dtype`` as little-endian, contiguous."""
    return np.ascontiguousarray(arr.astype(np.dtype(dtype).newbyteorder("<")))


def write_f32(path, arr):
    _le(arr, "<f4").tofile(str(path))


def write_f16(path, arr):
    _le(arr, "<f2").tofile(str(path))


def write_u8(path, arr):
    np.ascontiguousarray(arr.astype(np.uint8)).tofile(str(path))


def write_u16(path, arr):
    _le(arr, "<u2").tofile(str(path))


def write_i16(path, arr):
    _le(arr, "<i2").tofile(str(path))


def quantize_proba_u8(proba: np.ndarray) -> np.ndarray:
    """Softmax probabilities (n, 5) in [0, 1] -> uint8 (n, 5)."""
    return np.clip(np.rint(proba * 255.0), 0, 255).astype(np.uint8)
