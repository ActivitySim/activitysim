import numpy as np
import pytest


@pytest.fixture
def random_seed(request):
    current = np.random.get_state()

    def fin():
        np.random.set_state(current)
    request.addfinalizer(fin)

    np.random.seed(0)
