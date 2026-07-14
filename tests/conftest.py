import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_config_module():
    """Reloads config.py after every test so env-var/reload side effects
    from one test (e.g. test_config.py reloading with a patched .env)
    never leak into the next test, regardless of file."""
    yield
    import config as config_module
    importlib.reload(config_module)
