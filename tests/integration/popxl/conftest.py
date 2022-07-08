# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# See https://docs.pytest.org/en/6.2.x/fixture.html for details on this file.
import pytest
import numpy as np

SEED = 0


@pytest.fixture(autouse=True)
def random_seed():
    """Set the random seed for all tests in this directory. autouse=True will
    use this fixture in every test.
    """
    print(f"Setting numpy seed to {SEED}")
    np.random.seed(SEED)
