import numpy as np
import pytest

from avatarblend import default_ckpt, get_device, load_dataset, load_model
from traits import TraitSpace, build, traits_path


def _data_available():
    try:
        default_ckpt()
        return True
    except FileNotFoundError:
        return False


requires_model = pytest.mark.skipif(not _data_available(), reason="no trained checkpoint in checkpoints/")


@pytest.fixture(scope="session")
def model():
    if not _data_available():
        pytest.skip("no trained checkpoint in checkpoints/")
    return load_model(device=get_device())


@pytest.fixture(scope="session")
def data(model):
    return load_dataset(model.img_size)


@pytest.fixture(scope="session")
def traits_and_mu(model, data):
    from avatarblend import encode_mu
    if traits_path(model).exists():
        return TraitSpace.load(traits_path(model)), encode_mu(model, data["images"])
    return build(model, data)


@pytest.fixture(scope="session")
def residual_pool(traits_and_mu, data):
    from agent.tools import compute_residual_pool
    traits, mu = traits_and_mu
    return compute_residual_pool(traits, mu, data)


@pytest.fixture
def agent_ctx(model, traits_and_mu, residual_pool):
    from agent.tools import AgentContext
    traits, _ = traits_and_mu
    return AgentContext(model, traits, residual_pool, rng=np.random.default_rng(0))
