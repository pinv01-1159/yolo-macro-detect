"""
Chequeo de atajo (shortcut learning): ¿el FONDO ya predice la familia?

Entrena un clasificador trivial (centroide más cercano) sobre miniaturas de
32x32 en gris. A esa resolución el macroinvertebrado es un borrón: lo que
queda es fondo, encuadre e iluminación de la sesión de captura. Si el
accuracy supera al azar por mucho, el dataset tiene un confusor de sesión y
un mAP alto NO prueba que el modelo reconozca morfología.

Uso:
    python tools/confound_check.py datasets/clean
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def load(split_dir: Path, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for img in sorted((split_dir / "images").iterdir()):
        label = split_dir / "labels" / (img.stem + ".txt")
        lines = [ln for ln in label.read_text().split("\n") if ln.strip()]
        if not lines:
            continue
        a = np.asarray(
            Image.open(img).convert("L").resize((size, size), Image.BILINEAR), np.float32
        )
        a = ((a - a.mean()) / (a.std() + 1e-6)).ravel()
        xs.append(a / np.linalg.norm(a))
        ys.append(int(lines[0].split()[0]))
    return np.array(xs), np.array(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--eval-split", default="test")
    args = ap.parse_args()

    names = yaml.safe_load((args.dataset / "data.yaml").read_text())["names"]
    xtr, ytr = load(args.dataset / "train")
    xte, yte = load(args.dataset / args.eval_split)

    cent = np.array([xtr[ytr == c].mean(0) for c in range(len(names))])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    acc = float(((xte @ cent.T).argmax(1) == yte).mean())

    chance = 1 / len(names)
    majority = collections.Counter(yte).most_common(1)[0][1] / len(yte)
    print(f"accuracy solo-miniatura ({args.eval_split}): {acc:.1%}")
    print(f"  azar: {chance:.1%} | clase mayoritaria: {majority:.1%} | "
          f"ratio vs azar: {acc / chance:.1f}x")
    if acc > 3 * chance:
        print("  ⚠️  CONFUSOR DE SESIÓN: el fondo predice la clase. Reportalo como "
              "limitación y capturá cada familia en varias sesiones/fondos.")


if __name__ == "__main__":
    main()
