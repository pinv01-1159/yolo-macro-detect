"""
Dimensiones de las cajas de macroinvertebrados: estadística por familia y por
categoría de tamaño COCO, sobre el canvas real de cada imagen.

Categorías COCO (área en px^2 sobre la imagen de entrada al modelo):
    small  : area <  32^2 = 1024
    medium : 1024 <= area < 96^2 = 9216
    large  : area >= 9216

Salidas en --out:
    bbox_per_family.csv      una fila por familia (todas las instancias)
    bbox_per_split.csv       una fila por split
    bbox_instances.csv       una fila por instancia (dato crudo)
    bbox_summary.json        agregados + percentiles globales

Uso:
    python tools/audit_bbox_dimensions.py datasets/clean --out reports/paper
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SPLITS = ("train", "valid", "test")
SMALL = 32 * 32
MEDIUM = 96 * 96
PCTS = (5, 25, 50, 75, 95)


def coco_bucket(area_px: float) -> str:
    if area_px < SMALL:
        return "small"
    if area_px < MEDIUM:
        return "medium"
    return "large"


def pct(a: np.ndarray) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(a, p)) for p in PCTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/paper"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = yaml.safe_load((args.dataset / "data.yaml").read_text())["names"]
    rows = []
    canvas_sizes: defaultdict[str, int] = defaultdict(int)

    for split in SPLITS:
        img_dir, lbl_dir = args.dataset / split / "images", args.dataset / split / "labels"
        for img_path in sorted(img_dir.iterdir()):
            with Image.open(img_path) as im:
                W, H = im.size
            canvas_sizes[f"{W}x{H}"] += 1
            lbl = lbl_dir / f"{img_path.stem}.txt"
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            for ln in lines:
                cid, cx, cy, w, h = ln.split()
                cid = int(cid)
                w_px, h_px = float(w) * W, float(h) * H
                rows.append({
                    "split": split,
                    "image": img_path.stem,
                    "family": names[cid],
                    "img_w": W, "img_h": H,
                    "w_px": round(w_px, 2), "h_px": round(h_px, 2),
                    "area_px2": round(w_px * h_px, 2),
                    "area_frac": round(w_px * h_px / (W * H), 6),
                    "short_side_px": round(min(w_px, h_px), 2),
                    "long_side_px": round(max(w_px, h_px), 2),
                    "aspect_ratio": round(w_px / h_px, 4) if h_px > 0 else None,
                    "coco_size": coco_bucket(w_px * h_px),
                    "n_obj_in_image": len(lines),
                })

    with open(args.out / "bbox_instances.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    def block(sel: list[dict]) -> dict:
        area = np.array([r["area_px2"] for r in sel])
        frac = np.array([r["area_frac"] for r in sel])
        short = np.array([r["short_side_px"] for r in sel])
        long_ = np.array([r["long_side_px"] for r in sel])
        ar = np.array([r["aspect_ratio"] for r in sel])
        buckets = defaultdict(int)
        for r in sel:
            buckets[r["coco_size"]] += 1
        n = len(sel)
        return {
            "n": n,
            "area_px2_mean": float(area.mean()), "area_px2_median": float(np.median(area)),
            "area_frac_mean": float(frac.mean()), "area_frac_median": float(np.median(frac)),
            "short_side_px_mean": float(short.mean()),
            "short_side_px_median": float(np.median(short)),
            "long_side_px_median": float(np.median(long_)),
            "aspect_ratio_median": float(np.median(ar)),
            "pct_small": 100 * buckets["small"] / n,
            "pct_medium": 100 * buckets["medium"] / n,
            "pct_large": 100 * buckets["large"] / n,
            "n_small": buckets["small"], "n_medium": buckets["medium"],
            "n_large": buckets["large"],
            "short_side_pcts": pct(short),
            "area_px2_pcts": pct(area),
        }

    per_family = {fam: block([r for r in rows if r["family"] == fam])
                  for fam in names if any(r["family"] == fam for r in rows)}
    per_split = {s: block([r for r in rows if r["split"] == s]) for s in SPLITS}
    overall = block(rows)

    flat_fields = ["n", "area_px2_mean", "area_px2_median", "area_frac_mean", "area_frac_median",
                   "short_side_px_mean", "short_side_px_median", "long_side_px_median",
                   "aspect_ratio_median", "n_small", "n_medium", "n_large",
                   "pct_small", "pct_medium", "pct_large"]

    def dump_csv(path: Path, key_name: str, data: dict) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow([key_name, *flat_fields, *[f"short_p{p}" for p in PCTS]])
            for k, v in sorted(data.items(), key=lambda kv: kv[1]["area_px2_median"]):
                wtr.writerow([k, *[round(v[c], 4) if isinstance(v[c], float) else v[c]
                                   for c in flat_fields],
                              *[round(v["short_side_pcts"][f"p{p}"], 2) for p in PCTS]])

    dump_csv(args.out / "bbox_per_family.csv", "family", per_family)
    dump_csv(args.out / "bbox_per_split.csv", "split", per_split)

    summary = {
        "canvas_sizes": dict(canvas_sizes),
        "overall": overall,
        "per_split": per_split,
        "per_family": per_family,
        "n_instances": len(rows),
    }
    (args.out / "bbox_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"canvas: {dict(canvas_sizes)}")
    print(f"instancias: {len(rows)}")
    print(f"global: small {overall['pct_small']:.1f}% | medium {overall['pct_medium']:.1f}% | "
          f"large {overall['pct_large']:.1f}%")
    print(f"{'familia':<18}{'n':>5}{'área med':>10}{'%img':>8}{'ladoMin':>9}"
          f"{'%small':>8}{'%med':>8}{'%large':>8}")
    for fam, v in sorted(per_family.items(), key=lambda kv: kv[1]["area_px2_median"]):
        print(f"{fam:<18}{v['n']:>5}{v['area_px2_median']:>10.0f}"
              f"{100*v['area_frac_median']:>7.2f}%{v['short_side_px_median']:>9.1f}"
              f"{v['pct_small']:>7.1f}%{v['pct_medium']:>7.1f}%{v['pct_large']:>7.1f}%")


if __name__ == "__main__":
    main()
