"""Smoke tests: the package imports cleanly and the pretrained model runs a forward pass.

    pip install -e .
    pytest tests/                          # import graph only (no network)
    pytest tests/ --runslow                # also pull weights from HuggingFace

The import test guards the reproducibility contract: every entry-point module must
import against the pinned physioex==2.0.0. The forward-pass test is opt-in (network).
"""
import importlib
import pkgutil

import pytest


def test_import_all_modules():
    """Every protosleepnet submodule imports (scripts guard exec under __main__)."""
    import protosleepnet

    failures = {}
    for m in pkgutil.walk_packages(protosleepnet.__path__, "protosleepnet."):
        try:
            importlib.import_module(m.name)
        except Exception as e:  # noqa: BLE001
            failures[m.name] = repr(e)
    assert not failures, "modules failed to import:\n" + "\n".join(
        f"  {k}: {v}" for k, v in failures.items()
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,seq_len",
    [("protosleepnet-st-3ch-mixer", 21), ("protosleepnet-seq-3ch-mixer", 20)],
)
def test_pretrained_forward(name, seq_len):
    """Pretrained weights load from HF and produce (B, L, 5) AASM logits."""
    torch = pytest.importorskip("torch")
    from physioex.models import load_from_pretrained

    model = load_from_pretrained(name)
    x = torch.randn(2, seq_len, 3, 29, 129)  # (batch, L, channels, T, F)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, seq_len, 5)
