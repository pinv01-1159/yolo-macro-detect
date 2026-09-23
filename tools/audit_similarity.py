"""
¿El residual de coseno >0.90 entre valid/test y train es fuga remanente o el
piso natural de similitud del montaje de laboratorio?

Replica el descriptor de tools/build_clean_split.py (miniatura 32x32 gris,
normalizada por contraste, L2) y compara tres distribuciones de "máximo
coseno al vecino más cercano en train":

  (A) cross-split       : cada imagen de valid/test contra TODO train.
  (B) intra-train mismo grupo (piso superior): dentro de la misma ráfaga —
      es la similitud que genera la fuga real, el techo de referencia.
  (C) intra-train distinto grupo, MISMA clase (piso natural): la similitud
      que se obtiene entre dos especímenes distintos de la misma familia
      fotografiados en el mismo montaje. Si (A) ~ (C), el residual no es
      fuga: es el piso del dataset.
  (D) intra-train distinto grupo, OTRA clase (control negativo).

Uso:
    python tools/audit_similarity.py datasets/clean --out reports/paper
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SPLITS = ("train", "valid", "test")
SIZE = 32


def feats(paths: list[Path]) -> np.ndarray:
    f = np.zeros((len(paths), SIZE * SIZE), np.float32)
    for i, p in enumerate(paths):
        a = np.asarray(
            Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR), np.float32
        )
        f[i] = ((a - a.mean()) / (a.std() + 1e-6)).ravel()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def desc(a: np.ndarray) -> dict[str, float]:
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p5": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "frac_gt_0.90": float((a > 0.90).mean()),
        "frac_gt_0.95": float((a > 0.95).mean()),
        "frac_gt_0.99": float((a > 0.99).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/paper"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = yaml.safe_load((args.dataset / "data.yaml").read_text())["names"]
    gmap = json.loads((args.dataset / "split_report.json").read_text())["group_of_image"]

    paths: dict[str, list[Path]] = {}
    cls: dict[str, np.ndarray] = {}
    grp: dict[str, np.ndarray] = {}
    for s in SPLITS:
        ps = sorted((args.dataset / s / "images").iterdir())
        paths[s] = ps
        c, g = [], []
        for p in ps:
            ln = (args.dataset / s / "labels" / f"{p.stem}.txt").read_text().split("\n")[0]
            c.append(int(ln.split()[0]))
            g.append(gmap[p.stem])
        cls[s] = np.array(c)
        grp[s] = np.array(g)

    F = {s: feats(paths[s]) for s in SPLITS}
    out: dict = {
        "descriptor": "miniatura 32x32 gris, contraste-normalizado, L2 "
                      "(igual que build_clean_split.py)"
    }

    # (A) cross-split: max coseno de cada imagen de valid/test contra train
    for s in ("valid", "test"):
        sim = F[s] @ F["train"].T
        out[f"A_cross_{s}_vs_train"] = desc(sim.max(1))
        # desglose por clase
        per_cls = {}
        for c in range(len(names)):
            m = cls[s] == c
            if m.sum():
                per_cls[names[c]] = desc(sim[m].max(1))
        out[f"A_cross_{s}_vs_train_per_class"] = per_cls

    # intra-train
    St = F["train"] @ F["train"].T
    np.fill_diagonal(St, -1.0)
    same_g = grp["train"][:, None] == grp["train"][None, :]
    same_c = cls["train"][:, None] == cls["train"][None, :]

    def maxmask(mask: np.ndarray) -> np.ndarray:
        m = np.where(mask, St, -1.0)
        best = m.max(1)
        return best[best > -1.0]

    out["B_intra_train_same_group"] = desc(maxmask(same_g))
    out["C_intra_train_diff_group_same_class"] = desc(maxmask(~same_g & same_c))
    out["D_intra_train_diff_group_diff_class"] = desc(maxmask(~same_g & ~same_c))

    # C por clase (el piso natural por familia)
    per_cls_c = {}
    for c in range(len(names)):
        m = cls["train"] == c
        sub = np.where((~same_g & same_c)[np.ix_(m, m)], St[np.ix_(m, m)], -1.0)
        b = sub.max(1)
        b = b[b > -1.0]
        if b.size:
            per_cls_c[names[c]] = desc(b)
    out["C_per_class"] = per_cls_c

    (args.out / "audit_similarity.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    def line(k: str) -> None:
        d = out[k]
        print(f"{k:<42} n={d['n']:>5} med={d['median']:.3f} p95={d['p95']:.3f} "
              f">0.90={d['frac_gt_0.90']:.1%} >0.95={d['frac_gt_0.95']:.1%} "
              f">0.99={d['frac_gt_0.99']:.1%}")

    for k in ("A_cross_valid_vs_train", "A_cross_test_vs_train",
              "B_intra_train_same_group", "C_intra_train_diff_group_same_class",
              "D_intra_train_diff_group_diff_class"):
        line(k)


if __name__ == "__main__":
    main()
