import pytest

from sklearn.datasets import load_wine

from src.config import Config


@pytest.fixture(scope="session")
def wine():
    return load_wine()


@pytest.fixture()
def config():
    return Config()
