import importlib
import socket

import pytest


@pytest.fixture(autouse=True)
def _reset_config_module():
    """Reloads config.py after every test so env-var/reload side effects
    from one test (e.g. test_config.py reloading with a patched .env)
    never leak into the next test, regardless of file.

    Ojo: el reload crea una instancia NUEVA de `config`. Los modulos que
    hicieron `from config import config` siguen ligados a la anterior, asi que
    para parchear la configuracion que ve un modulo hay que parchearla en ese
    modulo (`mod.config`), no en `config.config`.
    """
    yield
    import config as config_module
    importlib.reload(config_module)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Corta el acceso a red durante los tests.

    Existe por un caso real: un test que creia estar probando el camino de
    "sin API key" parcheaba la instancia equivocada de config, pasaba la clave
    real y abria una conexion a Roboflow en cada corrida de CI. Un test que
    puede alcanzar un servicio externo con credenciales de verdad es un fallo
    de seguridad, no solo un test fragil.
    """
    def _bloqueado(*args, **kwargs):
        raise RuntimeError(
            "Acceso a red bloqueado en tests. Si el test necesita un servicio "
            "externo, mockealo; si necesita red de verdad, marcalo con "
            "@pytest.mark.network y desactiva esta fixture explicitamente."
        )

    monkeypatch.setattr(socket.socket, "connect", _bloqueado)
    monkeypatch.setattr(socket.socket, "connect_ex", _bloqueado)
    monkeypatch.setattr(socket, "create_connection", _bloqueado)
