import pytest

from glidertrack.configs import eval_config


@pytest.fixture
def short_two():
    return eval_config("two_glider", max_steps=200)


@pytest.fixture
def short_multi():
    return eval_config("multi_glider", n_gliders=4, max_steps=200)
