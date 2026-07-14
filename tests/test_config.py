import importlib

import config as config_module


def test_dotenv_file_is_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ROBOFLOW_API_KEY=from_dotenv_file\n")

    importlib.reload(config_module)

    assert config_module.config.roboflow_api_key == "from_dotenv_file"


def test_config_does_not_require_roboflow_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)

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
