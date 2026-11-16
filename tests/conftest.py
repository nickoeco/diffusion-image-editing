"""Keep real-weight and GPU checks explicit opt-ins, offline by default."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", help="Run marked CUDA tensor tests")
    parser.addoption("--run-integration", action="store_true", help="Run real model tests from existing local cache")
    parser.addoption("--run-demo", action="store_true", help="Run local HTTP demo and Gradio construction checks")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, option in (("gpu", "--run-gpu"), ("integration", "--run-integration"), ("demo", "--run-demo")):
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"requires explicit {option}"))
