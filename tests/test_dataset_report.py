import pytest
from PIL import Image

from reports.dataset_report import DatasetReport


def _make_image(path, size=(100, 100), color=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _make_label(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _build_dataset(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path) + "\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 2\n"
        "names: [Physidae, Chironomidae]\n"
    )

    _make_image(tmp_path / "train" / "images" / "img1.jpg")
    _make_label(
        tmp_path / "train" / "labels" / "img1.txt",
        ["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"],
    )

    _make_image(tmp_path / "train" / "images" / "img2.jpg")
    _make_label(tmp_path / "train" / "labels" / "img2.txt", ["0 0.5 0.5 0.2 0.2"])

    _make_image(tmp_path / "test" / "images" / "img3.jpg")
    _make_label(tmp_path / "test" / "labels" / "img3.txt", ["0 0.5 0.5 0.2 0.2"])

    return data_yaml


def test_generate_creates_output_files(tmp_path):
    data_yaml = _build_dataset(tmp_path)
    output_dir = tmp_path / "report_out"

    DatasetReport(data_yaml).generate(output_dir=output_dir)

    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "class_distribution.png").exists()


def test_class_distribution_counts(tmp_path):
    data_yaml = _build_dataset(tmp_path)

    report = DatasetReport(data_yaml).generate(output_dir=tmp_path / "report_out")

    assert report["splits"]["train"]["class_counts"]["Physidae"] == 2
    assert report["splits"]["train"]["class_counts"]["Chironomidae"] == 1
    assert report["splits"]["test"]["class_counts"]["Physidae"] == 1


def test_detects_malformed_box(tmp_path):
    data_yaml = _build_dataset(tmp_path)
    _make_label(tmp_path / "train" / "labels" / "img1.txt", ["0 0.5 0.5 0 0.2"])  # width = 0

    report = DatasetReport(data_yaml).generate(output_dir=tmp_path / "report_out")

    assert len(report["integrity_issues"]["malformed_boxes"]) >= 1


def test_detects_class_id_out_of_range(tmp_path):
    data_yaml = _build_dataset(tmp_path)
    _make_label(tmp_path / "train" / "labels" / "img1.txt", ["5 0.5 0.5 0.2 0.2"])

    report = DatasetReport(data_yaml).generate(output_dir=tmp_path / "report_out")

    assert len(report["integrity_issues"]["class_id_out_of_range"]) == 1


def test_detects_image_without_label(tmp_path):
    data_yaml = _build_dataset(tmp_path)
    _make_image(tmp_path / "train" / "images" / "orphan.jpg")

    report = DatasetReport(data_yaml).generate(output_dir=tmp_path / "report_out")

    assert any(
        "orphan.jpg" in item for item in report["integrity_issues"]["images_without_labels"]
    )


def test_detects_cross_split_duplicate(tmp_path):
    data_yaml = _build_dataset(tmp_path)
    # Copiar exactamente los mismos bytes de train/img1.jpg a test/ simula fuga de datos.
    source = tmp_path / "train" / "images" / "img1.jpg"
    duplicate = tmp_path / "test" / "images" / "img1_dup.jpg"
    duplicate.write_bytes(source.read_bytes())
    _make_label(tmp_path / "test" / "labels" / "img1_dup.txt", ["0 0.5 0.5 0.2 0.2"])

    report = DatasetReport(data_yaml).generate(output_dir=tmp_path / "report_out")

    assert len(report["duplicate_leakage"]) == 1
    leak = report["duplicate_leakage"][0]
    assert set(leak["splits"]) == {"train", "test"}


def test_duplicate_leakage_caveat_present_in_json_and_markdown(tmp_path):
    data_yaml = _build_dataset(tmp_path)
    output_dir = tmp_path / "report_out"

    report = DatasetReport(data_yaml).generate(output_dir=output_dir)

    assert "duplicate_leakage_caveat" in report
    caveat = report["duplicate_leakage_caveat"]
    assert "MD5" in caveat
    assert "aumenta" in caveat.lower()  # "aumentación" / "aumentadas"

    md_content = (output_dir / "report.md").read_text(encoding="utf-8")
    assert caveat in md_content


def test_bbox_aspect_ratio_uses_pixel_dimensions_not_normalized(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path) + "\n"
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 1\n"
        "names: [Physidae]\n"
    )
    # Non-square image (200x100). Box: normalized w=0.2, h=0.4 =>
    # pixel width = 0.2*200 = 40, pixel height = 0.4*100 = 40 => square (true
    # ratio 1.0). Normalized ratio would be 0.2/0.4 = 0.5 -- wrong.
    _make_image(tmp_path / "train" / "images" / "img1.jpg", size=(200, 100))
    _make_label(tmp_path / "train" / "labels" / "img1.txt", ["0 0.5 0.5 0.2 0.4"])

    report = DatasetReport(data_yaml).generate(output_dir=tmp_path / "report_out")

    aspect = report["splits"]["train"]["bbox_aspect_ratio"]
    assert aspect["mean"] == pytest.approx(1.0)
