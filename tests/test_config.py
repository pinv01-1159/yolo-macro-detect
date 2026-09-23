import importlib

import config as config_module


def test_config_no_depende_del_directorio_de_trabajo(tmp_path, monkeypatch):
    """El .env se lee junto al código, no desde donde se invoca el programa.

    Antes se leía de `Path.cwd()`, así que correr el pipeline desde otra
    carpeta cambiaba la configuración en silencio: bastaba con que hubiera un
    `.env` ajeno en el directorio actual para secuestrar las credenciales y los
    hiperparámetros.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.delenv("SEED", raising=False)
    (tmp_path / ".env").write_text(
        "ROBOFLOW_API_KEY=clave_de_otro_directorio\nSEED=999\n"
    )

    importlib.reload(config_module)

    assert config_module.config.roboflow_api_key != "clave_de_otro_directorio"
    assert config_module.config.seed != 999


def test_una_variable_de_entorno_gana_sobre_el_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "desde_el_entorno")

    importlib.reload(config_module)

    assert config_module.config.roboflow_api_key == "desde_el_entorno"


def test_config_does_not_require_roboflow_api_key(tmp_path, monkeypatch):
    """Sin clave la configuración sigue siendo válida: no toda operación la
    necesita (p. ej. --predict). La exige quien se conecta a Roboflow."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "")

    importlib.reload(config_module)

    assert config_module.config.roboflow_api_key == ""
    assert config_module.config.validate() is True


def test_validate_rejects_bad_confidence_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.5")

    importlib.reload(config_module)

    assert config_module.config.validate() is False


def test_seed_defaults_to_42(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SEED", raising=False)

    importlib.reload(config_module)

    assert config_module.config.seed == 42


def test_img_size_debe_ser_multiplo_de_32(tmp_path, monkeypatch):
    """YOLO reescala en silencio si el lado no es múltiplo del stride máximo:
    la resolución efectiva deja de ser la declarada y nadie se entera."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMG_SIZE", "641")

    importlib.reload(config_module)

    assert config_module.config.validate() is False
    assert any("multiplo de 32" in e for e in config_module.config.validation_errors())


def test_un_entero_vacio_no_revienta_el_arranque(tmp_path, monkeypatch):
    """`int(os.getenv(...))` moría con un ValueError crudo en el constructor,
    antes de que validate() pudiera decir qué estaba mal."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMG_SIZE", "")
    monkeypatch.setenv("BATCH_SIZE", "no-es-un-numero")

    importlib.reload(config_module)  # no debe lanzar

    assert config_module.config.img_size == 640
    assert config_module.config.batch_size == 16
    avisos = " ".join(config_module.config.validation_errors())
    assert "IMG_SIZE" in avisos and "BATCH_SIZE" in avisos


def test_validation_errors_nombra_el_problema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.5")

    importlib.reload(config_module)

    assert config_module.config.validate() is False
    assert any("CONFIDENCE_THRESHOLD" in e for e in config_module.config.validation_errors())
