import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run slow tests that pull weights from HuggingFace (network required)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow (network)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
