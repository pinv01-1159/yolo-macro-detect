"""
Cruza la geometría de las cajas por familia (audit_bbox_dimensions.py) contra
el desempeño por familia (audit_eval_checkpoints.py) y calcula correlaciones.

Responde: ¿el desempeño por familia se explica por el tamaño de la caja?
Se reportan Pearson y Spearman de mAP50-95 vs. (área mediana, fracción de
cuadro, lado menor, elongación), por modelo y promediando los tres.

Salida: reports/paper/family_size_vs_performance.csv + resumen por stdout.

Uso:
    python tools/audit_size_vs_performance.py --out reports/paper
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    # promediar empates
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pear = float(np.corrcoef(x, y)[0, 1])
    spear = float(np.corrcoef(rank(x), rank(y))[0, 1])
    return pear, spear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/paper"))
    ap.add_argument("--protocol", default="coco")
    args = ap.parse_args()

    geo = {r["family"]: r for r in csv.DictReader(open(args.out / "bbox_per_family.csv"))}

    # elongacion correcta: mediana_i( max(AR_i, 1/AR_i) ), calculada sobre instancias
    inst = defaultdict(list)
    for r in csv.DictReader(open(args.out / "bbox_instances.csv")):
        ar = float(r["aspect_ratio"])
        inst[r["family"]].append(max(ar, 1 / ar))
    elong = {f: float(np.median(v)) for f, v in inst.items()}
    perf = defaultdict(dict)
    for r in csv.DictReader(open(args.out / "eval_per_class.csv")):
        if r["protocol"] == args.protocol:
            perf[r["family"]][r["model"]] = r

    models = sorted({m for v in perf.values() for m in v})
    fams = sorted(geo)

    rows = []
    for f in fams:
        g = geo[f]
        row = {
            "familia": f,
            "n_instancias": int(g["n"]),
            "area_mediana_px2": round(float(g["area_px2_median"])),
            "frac_cuadro_mediana": round(float(g["area_frac_median"]), 4),
            "lado_menor_mediana_px": round(float(g["short_side_px_median"]), 1),
            "aspect_ratio_mediana": round(float(g["aspect_ratio_median"]), 3),
            # OJO: mediana de max(AR,1/AR) POR INSTANCIA, no max() de la mediana.
            # Agregar antes de aplicar max() subestima a los taxones que aparecen
            # en ambas orientaciones (p. ej. Hirudinidae: 1.79 mal vs 2.17 bien).
            "elongacion_mediana": round(elong[f], 3),
            "pct_small_coco": float(g["pct_small"]),
            "pct_medium_coco": float(g["pct_medium"]),
            "pct_large_coco": float(g["pct_large"]),
        }
        aps = []
        for m in models:
            p = perf[f].get(m)
            if not p:
                continue
            for k in ("precision", "recall", "f1", "map50", "map50_95"):
                row[f"{k}_{m}"] = round(float(p[k]), 4)
            aps.append(float(p["map50_95"]))
        row["map50_95_mean"] = round(float(np.mean(aps)), 4) if aps else None
        rows.append(row)

    with open(args.out / "family_size_vs_performance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    y = np.array([r["map50_95_mean"] for r in rows], float)
    preds = {
        "área mediana (px²)": np.array([r["area_mediana_px2"] for r in rows], float),
        "fracción del cuadro": np.array([r["frac_cuadro_mediana"] for r in rows], float),
        "lado menor (px)": np.array([r["lado_menor_mediana_px"] for r in rows], float),
        "elongación mediana(max(AR,1/AR))": np.array(
            [r["elongacion_mediana"] for r in rows], float
        ),
        "n instancias": np.array([r["n_instancias"] for r in rows], float),
    }

    print(f"protocolo={args.protocol} · n familias={len(rows)} · modelos={models}")
    print(f"mAP50-95 medio por familia: min={y.min():.3f} max={y.max():.3f} media={y.mean():.3f}")
    print(f"\n{'predictor':<28}{'Pearson r':>11}{'Spearman ρ':>12}")
    for name, x in preds.items():
        p, s = corr(x, y)
        print(f"{name:<28}{p:>11.3f}{s:>12.3f}")

    print(f"\n{'familia':<18}{'mAP50-95':>10}{'área med':>10}{'%cuadro':>9}{'elong':>7}")
    for r in sorted(rows, key=lambda r: r["map50_95_mean"]):
        print(f"{r['familia']:<18}{r['map50_95_mean']:>10.3f}{r['area_mediana_px2']:>10}"
              f"{100*r['frac_cuadro_mediana']:>8.1f}%{r['elongacion_mediana']:>7.2f}")


if __name__ == "__main__":
    main()
