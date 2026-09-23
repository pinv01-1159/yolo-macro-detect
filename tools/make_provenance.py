"""Genera reports/paper/provenance.json: de donde sale cada cifra del informe.

La cadena "numero publicado -> artefacto" ya estaba cubierta por los CSV. Lo
que faltaba era el otro extremo: con que commit, que pesos y que comando se
produjo cada artefacto. Sin eso un tercero puede leer los numeros pero no
puede rehacerlos, y la columna `save_dir` de algunos CSV apunta a directorios
de trabajo efimeros que no existiran para nadie mas.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "reports" / "paper"

# Cada artefacto con el comando que lo regenera. Si un artefacto no puede
# declarar como se rehace, no es evidencia: es un archivo.
COMANDOS = {
    "audit_splits.json": "uv run python tools/audit_splits.py",
    "groups_per_class.csv": "uv run python tools/audit_splits.py",
    "audit_similarity.json": "uv run python tools/audit_similarity.py",
    "bbox_per_family.csv": "uv run python tools/audit_bbox_dimensions.py",
    "bbox_per_split.csv": "uv run python tools/audit_bbox_dimensions.py",
    "bbox_instances.csv": "uv run python tools/audit_bbox_dimensions.py",
    "bbox_summary.json": "uv run python tools/audit_bbox_dimensions.py",
    "eval_overall.csv": "uv run python tools/audit_eval_checkpoints.py",
    "eval_overall.json": "uv run python tools/audit_eval_checkpoints.py",
    "eval_per_class.csv": "uv run python tools/audit_eval_checkpoints.py",
    "family_size_vs_performance.csv": "uv run python tools/audit_size_vs_performance.py",
    "threshold_sweep_valid.csv": "uv run python tools/audit_sweep_and_bootstrap.py",
    "bootstrap_grouped.json": "uv run python tools/audit_sweep_and_bootstrap.py",
}
for m in ("yolo11s", "yolo12s", "yolo26s"):
    for suf in ("abs", "norm"):
        COMANDOS[f"confusion_{m}_{suf}.csv"] = "uv run python tools/audit_eval_checkpoints.py"

FIGURAS = {
    "informe/imagenes/elongacion_vs_map.png": "uv run python tools/make_report_figures.py",
    "informe/imagenes/distribucion_tamanos_bbox.png": "uv run python tools/make_report_figures.py",
    "informe/imagenes/ejemplo_simple.png": "uv run python tools/make_inference_examples.py",
    "informe/imagenes/ejemplo_multiple.png": "uv run python tools/make_inference_examples.py",
    "informe/imagenes/ejemplo_dificil.png": "uv run python tools/make_inference_examples.py",
    "informe/imagenes/confusion_matrix_normalized.png":
        "cp reports/paper/val_runs/yolo26s_coco/confusion_matrix_normalized.png",
    "informe/imagenes/pipeline_validacion.jpg":
        "cp reports/paper/val_runs/yolo26s_coco/val_batch0_pred.jpg",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def entrada(path: Path, comando: str | None) -> dict:
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "regenerar_con": comando,
    }


def main() -> None:
    limpio = git("status", "--porcelain") == ""
    doc = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "arbol_limpio": limpio,
            "aviso": None if limpio else (
                "El arbol tenia cambios sin commitear al generar este archivo: "
                "el commit de arriba NO describe el estado que produjo los "
                "artefactos. Regenerar despues de commitear."
            ),
        },
        "checkpoints": {},
        "dataset": {},
        "artefactos": {},
        "figuras": {},
    }

    for pt in sorted((ROOT / "checkpoints").glob("*.pt")):
        doc["checkpoints"][pt.name] = {
            "sha256": sha256(pt),
            "bytes": pt.stat().st_size,
        }

    split_report = ROOT / "datasets" / "clean" / "split_report.json"
    if split_report.exists():
        doc["dataset"] = {
            "split_report_sha256": sha256(split_report),
            "regenerar_con": (
                "uv run python tools/build_clean_split.py datasets/v9 datasets/clean"
            ),
            "roboflow": "pinv011159/macroinvertebrados-split-limpio-kuhq3",
        }

    for nombre, comando in sorted(COMANDOS.items()):
        ruta = PAPER / nombre
        if ruta.exists():
            doc["artefactos"][nombre] = entrada(ruta, comando)

    for rel, comando in sorted(FIGURAS.items()):
        ruta = ROOT / rel
        if ruta.exists():
            doc["figuras"][rel] = entrada(ruta, comando)

    destino = PAPER / "provenance.json"
    destino.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    faltan = [n for n in COMANDOS if not (PAPER / n).exists()]
    assert not faltan, f"artefactos declarados que no existen: {faltan}"
    print(f"escrito: {destino}")
    print(f"  commit {doc['git']['commit'][:8]} · arbol_limpio={limpio}")
    print(f"  {len(doc['checkpoints'])} checkpoints · "
          f"{len(doc['artefactos'])} artefactos · {len(doc['figuras'])} figuras")


if __name__ == "__main__":
    main()
