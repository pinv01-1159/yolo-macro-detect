from unittest.mock import MagicMock

import pytest

import data.dataset_manager as dataset_manager_module
from data.dataset_manager import DatasetManager


def test_dataset_manager_init_does_not_require_roboflow_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)

    manager = DatasetManager()

    assert manager.rf is None
    assert manager.project is None


def test_setup_roboflow_connection_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Hay que parchear el objeto que dataset_manager tiene ligado, no el de
    # config: el modulo hace `from config import config` al importarse, asi que
    # el reload de conftest deja dos instancias distintas y parchear la de
    # config no afecta a la que se usa aca. Con la instancia equivocada el test
    # pasaba la clave real y golpeaba la API de Roboflow.
    monkeypatch.setattr(dataset_manager_module.config, "roboflow_api_key", "")
    manager = DatasetManager()

    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        manager.setup_roboflow_connection()


def test_setup_roboflow_connection_succeeds_with_mocked_roboflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_roboflow = MagicMock()
    fake_project = MagicMock()
    fake_roboflow.return_value.workspace.return_value.project.return_value = fake_project
    monkeypatch.setattr("data.dataset_manager.Roboflow", fake_roboflow)

    manager = DatasetManager()
    manager.setup_roboflow_connection(api_key="fake-key", workspace="ws", project_name="proj")

    fake_roboflow.assert_called_once_with(api_key="fake-key")
    assert manager.project is fake_project
