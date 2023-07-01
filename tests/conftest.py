"""The conftest.py file allows us to initialise test functions
that can be repeatedly used across several tests.
"""

import pickle
from pathlib import Path

import pytest

from cybersyn import Economy


@pytest.fixture
def spanish_economy() -> Economy:
    """Return the spanish economy in an Economy object."""
    with Path("data", "spanish_economy.pkl").open("rb") as f:
        economy_dict = pickle.load(f)
    return Economy(**economy_dict)
