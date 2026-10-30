import pytest

from sklearn.datasets import load_wine


@pytest.fixture(scope="session")
def wine():
    return load_wine()
