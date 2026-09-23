"""
Auditoría independiente de datasets/clean: integridad de splits y etiquetas.

Verifica CONTRA LOS ARCHIVOS EN DISCO (no contra split_report.json):
  1. Ningún group_id aparece en más de un split.
  2. Toda imagen en disco está cubierta por group_of_image.
  3. Conteos de imágenes/instancias por clase coinciden con el JSON.
  4. Sanidad de etiquetas: .txt faltante/vacío, coords fuera de [0,1],
     cajas degeneradas, class_id fuera de rango, huérfanos en ambos sentidos.
  5. Contaminación por augmentación: presencia de sufijo .rf.<hash> por split.

Uso:
    python tools/audit_splits.py datasets/clean --out reports/paper
"""

import argparse
import collections
import json
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/paper"))
    args = ap.parse_args()

    names = yaml.safe_load((args.dataset / "data.yaml").read_text())["names"]
    report = json.loads((args.dataset / "split_report.json").read_text())
    group_of_image = report["group_of_image"]

    out: dict = {"dataset": str(args.dataset), "n_classes": len(names)}

    # --- 1 y 2: grupos vs splits, a partir del disco ---
    split_of_stem: dict[str, str] = {}
    groups_by_split: dict[str, set[int]] = {s: set() for s in SPLITS}
    uncovered: list[str] = []
    for s in SPLITS:
        for img in sorted((args.dataset / s / "images").iterdir()):
            stem = img.stem
            split_of_stem[stem] = s
            if stem not in group_of_image:
                uncovered.append(f"{s}/{img.name}")
            else:
                groups_by_split[s].add(group_of_image[stem])

    cross = []
    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            sa, sb = SPLITS[a], SPLITS[b]
            shared = groups_by_split[sa] & groups_by_split[sb]
            if shared:
                cross.append({"splits": [sa, sb], "n_groups": len(shared),
                              "groups": sorted(shared)[:20]})

    orphan_json = sorted(set(group_of_image) - set(split_of_stem))
    out["group_integrity"] = {
        "groups_per_split": {s: len(groups_by_split[s]) for s in SPLITS},
        "groups_total_disk": len(set().union(*groups_by_split.values())),
        "groups_in_json": report["groups"],
        "cross_split_group_collisions": cross,
        "images_on_disk_not_in_group_of_image": uncovered,
        "entries_in_json_not_on_disk": orphan_json,
    }

    # --- 3, 4: integridad de etiquetas y conteos reales ---
    counts: dict[str, collections.Counter] = {s: collections.Counter() for s in SPLITS}
    issues: dict[str, list] = {
        "image_without_label": [], "empty_label": [], "label_without_image": [],
        "coord_out_of_range": [], "degenerate_box": [], "class_id_out_of_range": [],
        "malformed_line": [], "box_exceeds_canvas": [],
    }
    objs_per_image: dict[str, list[int]] = {s: [] for s in SPLITS}
    rf_suffix_per_split: dict[str, int] = {s: 0 for s in SPLITS}

    for s in SPLITS:
        img_dir, lbl_dir = args.dataset / s / "images", args.dataset / s / "labels"
        stems_img = {p.stem for p in img_dir.iterdir()}
        stems_lbl = {p.stem for p in lbl_dir.glob("*.txt")}
        rf_suffix_per_split[s] = sum(1 for st in stems_img if ".rf." in st)
        for st in sorted(stems_img - stems_lbl):
            issues["image_without_label"].append(f"{s}/{st}")
        for st in sorted(stems_lbl - stems_img):
            issues["label_without_image"].append(f"{s}/{st}")

        for st in sorted(stems_img & stems_lbl):
            lines = [ln for ln in (lbl_dir / f"{st}.txt").read_text().splitlines() if ln.strip()]
            if not lines:
                issues["empty_label"].append(f"{s}/{st}")
            n = 0
            for i, ln in enumerate(lines, 1):
                parts = ln.split()
                if len(parts) != 5:
                    issues["malformed_line"].append(f"{s}/{st}:{i}")
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    issues["malformed_line"].append(f"{s}/{st}:{i}")
                    continue
                if not (0 <= cid < len(names)):
                    issues["class_id_out_of_range"].append(f"{s}/{st}:{i} cid={cid}")
                    continue
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    issues["coord_out_of_range"].append(f"{s}/{st}:{i} {cx},{cy},{w},{h}")
                if w <= 1e-6 or h <= 1e-6:
                    issues["degenerate_box"].append(f"{s}/{st}:{i} w={w} h={h}")
                if cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
                        or cy + h / 2 > 1 + 1e-6:
                    issues["box_exceeds_canvas"].append(f"{s}/{st}:{i}")
                counts[s][names[cid]] += 1
                n += 1
            objs_per_image[s].append(n)

    out["label_integrity"] = {k: {"n": len(v), "examples": v[:10]} for k, v in issues.items()}
    out["instances_per_split_disk"] = {s: dict(sorted(counts[s].items())) for s in SPLITS}
    out["images_per_split_disk"] = {
        s: len(list((args.dataset / s / "images").iterdir())) for s in SPLITS
    }
    out["objects_per_image"] = {
        s: {"total_images": len(objs_per_image[s]),
            "with_1": sum(1 for v in objs_per_image[s] if v == 1),
            "with_gt1": sum(1 for v in objs_per_image[s] if v > 1),
            "with_0": sum(1 for v in objs_per_image[s] if v == 0),
            "max": max(objs_per_image[s]) if objs_per_image[s] else 0}
        for s in SPLITS
    }
    out["roboflow_aug_suffix_count"] = rf_suffix_per_split

    # --- diff contra el JSON ---
    diffs = []
    for s in SPLITS:
        j = report["instances_per_split"][s]
        for n in names:
            if j.get(n, 0) != counts[s][n]:
                diffs.append({"split": s, "class": n, "json": j.get(n, 0), "disk": counts[s][n]})
        if report["images_per_split"][s] != out["images_per_split_disk"][s]:
            diffs.append({"split": s, "class": "__images__",
                          "json": report["images_per_split"][s],
                          "disk": out["images_per_split_disk"][s]})
    out["json_vs_disk_diffs"] = diffs

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit_splits.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("instances_per_split_disk",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
