import pytest

from utils.validators import (
    validate_confidence_threshold,
    validate_image_path,
    validate_iou_threshold,
    validate_model_path,
)


def test_validate_image_path_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_image_path(tmp_path / "missing.jpg")


def test_validate_image_path_wrong_extension(tmp_path):
    bad_file = tmp_path / "photo.txt"
    bad_file.write_text("not an image")

    with pytest.raises(ValueError, match="Extensión"):
        validate_image_path(bad_file)


def test_validate_image_path_does_not_decode_the_image(tmp_path):
    # Contenido inválido como JPEG: si el validador intentara decodificarla
    # (como hacía antes), esto fallaría. Ahora solo valida existencia/extensión.
    fake_image = tmp_path / "photo.jpg"
    fake_image.write_bytes(b"not a real jpeg")

    assert validate_image_path(fake_image) is True


def test_validate_model_path_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_model_path(tmp_path / "missing.pt")


def test_validate_model_path_wrong_extension(tmp_path):
    bad_file = tmp_path / "model.bin"
    bad_file.write_text("not a model")

    with pytest.raises(ValueError, match="extensión .pt"):
        validate_model_path(bad_file)


def test_validate_model_path_does_not_load_the_model(tmp_path):
    # Contenido inválido como checkpoint: si el validador intentara cargarlo
    # con YOLO(...) (como hacía antes), esto fallaría. Ahora solo valida
    # existencia/extensión.
    fake_model = tmp_path / "model.pt"
    fake_model.write_bytes(b"not a real checkpoint")

    assert validate_model_path(fake_model) is True


def test_validate_confidence_threshold_valid():
    assert validate_confidence_threshold(0.5) is True


def test_validate_confidence_threshold_out_of_range():
    with pytest.raises(ValueError):
        validate_confidence_threshold(1.5)


def test_validate_iou_threshold_valid():
    assert validate_iou_threshold(0.6) is True


def test_validate_iou_threshold_out_of_range():
    with pytest.raises(ValueError):
        validate_iou_threshold(-0.1)
