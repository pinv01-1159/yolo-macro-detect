"""
Sube un dataset YOLO local a un proyecto nuevo de Roboflow, respetando el split.

Reanudable: relee lo ya subido y saltea esas imágenes, así que se puede
volver a correr si se corta la red.

Uso:
    python tools/upload_to_roboflow.py datasets/clean "Macroinvertebrados Split Limpio"
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

SPLITS = ("train", "valid", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("project_name")
    ap.add_argument("--workspace", default="pinv011159")
    args = ap.parse_args()

    load_dotenv(Path.cwd() / ".env")
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ROBOFLOW_API_KEY no configurado")

    names = yaml.safe_load((args.dataset / "data.yaml").read_text())["names"]
    labelmap = dict(enumerate(names))

    rf = Roboflow(api_key=key)
    ws = rf.workspace(args.workspace)

    slug = args.project_name.lower().replace(" ", "-")
    try:
        project = ws.project(slug)
        print(f"proyecto existente: {slug}")
    except Exception:
        project = ws.create_project(
            project_name=args.project_name,
            project_type="object-detection",
            project_license="MIT",
            annotation="macroinvertebrados",
        )
        print(f"proyecto creado: {project.id}")

    done: set[str] = set()
    while True:  # search pagina de a 100
        page = project.search(limit=100, offset=len(done), fields=["name"])
        if not page:
            break
        done.update(img["name"] for img in page)
    print(f"{len(done)} imágenes ya presentes, se saltean")

    ok = fail = 0
    for split in SPLITS:
        for img in sorted((args.dataset / split / "images").iterdir()):
            if img.name in done:
                continue
            try:
                project.single_upload(
                    image_path=str(img),
                    annotation_path=str(args.dataset / split / "labels" / f"{img.stem}.txt"),
                    annotation_labelmap=labelmap,
                    split=split,
                    num_retry_uploads=3,
                )
                ok += 1
            except Exception as e:  # noqa: BLE001 - una imagen mala no debe cortar la subida
                fail += 1
                print(f"  fallo {img.name}: {e}", flush=True)
            if (ok + fail) % 50 == 0:
                print(f"  {split}: {ok} ok, {fail} fallos", flush=True)
                # ponytail: cortesía con el rate limit; pasar a backoff si aparece un 429
                time.sleep(1)
    print(f"listo: {ok} subidas, {fail} fallos")


if __name__ == "__main__":
    main()
