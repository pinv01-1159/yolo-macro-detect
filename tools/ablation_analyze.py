"""Analisis offline de raw_<modelo>.json -> summary_<modelo>.json + tablas.

Criterios:
  crop_acc        : clase de mayor conf en el recorte == clase GT (identico al script).
  bg_orig         : identico al script: primera deteccion (orden conf desc) cuyo
                    centro esta a < max(bw,bh) del centro GT; acierto si clase == GT.
  bg_clean        : igual, pero descartando detecciones que corresponden a OTRO
                    organismo real de la imagen (IoU>=0.3 con otra caja GT o centro
                    dentro de otra caja GT). Corrige el defecto de multi-instancia.
  full_hit        : en la imagen original, existe prediccion con IoU>=0.5 con la caja
                    GT y misma clase (referencia con contexto completo).
Azar (bg): tasa_deteccion_en_zona * sum_c P(pred=c | det) * P(gt=c | det)
           (independencia clase-predicha/clase-real dado que hubo deteccion), y 1/19.
Azar (crop): 1/19 y clase mayoritaria del subconjunto.
IC 95%: bootstrap por especimen (group_of_image de split_report.json), 2000 remuestras.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NAMES = [
    "Ampullariidae",
    "Ancylidae",
    "Belostomatidae",
    "Ceratopogonidae",
    "Chironomidae",
    "Coenagrionidae",
    "Dytiscidae",
    "Gerridae",
    "Glossiphoniidae",
    "Hirudinidae",
    "Hydrophilidae",
    "Hyriidae",
    "Libellulidae",
    "Miridae",
    "Noteridae",
    "Notonectidae",
    "Physidae",
    "Planorbidae",
    "Psychodidae",
]
G = json.load(open(REPO / "datasets/clean/split_report.json"))["group_of_image"]
B = 2000
rng = np.random.default_rng(0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def bg_eval(r, others, clean):
    box = r["box"]
    rad = max(box[2] - box[0], box[3] - box[1])
    gc = center(box)
    for d in r["bg"]:  # ultralytics devuelve ordenado por conf desc
        pc = center(d["xyxy"])
        if ((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2) ** 0.5 >= rad:
            continue
        if clean and any(
            iou(d["xyxy"], o) >= 0.3 or (o[0] <= pc[0] <= o[2] and o[1] <= pc[1] <= o[3])
            for o in others
        ):
            continue
        return True, d["cls"]
    return False, None


def per_instance(raw):
    recs = raw["records"]
    by_img = defaultdict(list)
    for r in recs:
        by_img[r["image"]].append(r)
    rows = []
    for r in recs:
        others = [o["box"] for o in by_img[r["image"]] if o is not r]
        same_cls_others = sum(
            o["class_id"] == r["class_id"] for o in by_img[r["image"]] if o is not r
        )
        crop_pred = max(r["crop"], key=lambda d: d["conf"])["cls"] if r["crop"] else None
        det_o, cls_o = bg_eval(r, others, False)
        det_c, cls_c = bg_eval(r, others, True)
        full_hit = any(
            d["cls"] == r["class_id"] and iou(d["xyxy"], r["box"]) >= 0.5 for d in r["full"]
        )
        rows.append(
            dict(
                image=r["image"],
                group=G[r["image"]],
                y=r["class_id"],
                area=r["area_frac"],
                multi=len(others) > 0,
                same_cls_others=same_cls_others,
                crop_ok=crop_pred == r["class_id"],
                crop_nodet=crop_pred is None,
                bg_det_o=det_o,
                bg_cls_o=cls_o,
                bg_ok_o=det_o and cls_o == r["class_id"],
                bg_det_c=det_c,
                bg_cls_c=cls_c,
                bg_ok_c=det_c and cls_c == r["class_id"],
                full_hit=full_hit,
            )
        )
    return rows


def chance_bg(rows, det_key, cls_key):
    d = [r for r in rows if r[det_key]]
    if not d:
        return 0.0
    pp = np.bincount([r[cls_key] for r in d], minlength=19) / len(d)
    pt = np.bincount([r["y"] for r in d], minlength=19) / len(d)
    return len(d) / len(rows) * float(pp @ pt)


def stats(rows):
    if not rows:
        return {"n": 0}
    m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    ys = np.bincount([r["y"] for r in rows], minlength=19)
    return {
        "n": len(rows),
        "n_specimens": len({r["group"] for r in rows}),
        "n_images": len({r["image"] for r in rows}),
        "full_hit": m("full_hit"),
        "crop_acc": m("crop_ok"),
        "crop_nodet": m("crop_nodet"),
        "bg_shortcut_orig": m("bg_ok_o"),
        "bg_anydet_orig": m("bg_det_o"),
        "bg_shortcut_clean": m("bg_ok_c"),
        "bg_anydet_clean": m("bg_det_c"),
        "bg_chance_orig": chance_bg(rows, "bg_det_o", "bg_cls_o"),
        "bg_chance_clean": chance_bg(rows, "bg_det_c", "bg_cls_c"),
        "crop_chance_uniform": 1 / 19,
        "crop_chance_majority": float(ys.max() / len(rows)),
        "frac_multi_instance": m("multi"),
    }


KEYS = ["full_hit", "crop_acc", "bg_shortcut_orig", "bg_shortcut_clean", "bg_chance_clean"]


def boot(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    gl = list(groups.values())
    if len(gl) < 2:
        return {}
    acc = defaultdict(list)
    for _ in range(B):
        sample = [r for i in rng.integers(0, len(gl), len(gl)) for r in gl[i]]
        s = stats(sample)
        for k in KEYS:
            acc[k].append(s[k])
        acc["bg_excess_clean"].append(s["bg_shortcut_clean"] - s["bg_chance_clean"])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()
    }


def block(rows):
    s = stats(rows)
    s["ci95_cluster"] = boot(rows)
    return s


def main(model):
    d = REPO / "reports/paper/ablation"
    raw = json.load(open(d / f"raw_{model}.json"))
    rows = per_instance(raw)
    small = [r for r in rows if r["area"] < 0.40]
    out = {
        "model": raw["model"],
        "conf": raw["conf"],
        "seconds": raw["seconds"],
        "all": block(rows),
        "small_lt_0.40": block(small),
        "small_lt_0.40_single_instance": block([r for r in small if not r["multi"]]),
        "per_family": {},
    }
    for c, name in enumerate(NAMES):
        out["per_family"][name] = {
            "all": stats([r for r in rows if r["y"] == c]),
            "small": stats([r for r in small if r["y"] == c]),
            "median_area": float(np.median([r["area"] for r in rows if r["y"] == c] or [np.nan])),
        }
    # cuantas veces el criterio original cuenta como "atajo" a otro organismo real
    out["orig_hits_explained_by_other_gt"] = int(
        sum(r["bg_ok_o"] and not r["bg_ok_c"] for r in rows)
    )
    (d / f"summary_{model}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (d / f"rows_{model}.json").write_text(json.dumps(rows))
    f = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    ci = lambda b, k: (  # noqa: E731
        f"[{f(b['ci95_cluster'][k][0])},{f(b['ci95_cluster'][k][1])}]" if b["ci95_cluster"] else ""
    )
    for lab in ["all", "small_lt_0.40", "small_lt_0.40_single_instance"]:
        b = out[lab]
        if not b["n"]:
            print(f"\n== {model} {lab}: n=0")
            continue
        print(f"\n== {model} {lab}: n={b['n']} esp={b['n_specimens']} img={b['n_images']}")
        for k in [
            "full_hit",
            "crop_acc",
            "crop_nodet",
            "bg_shortcut_orig",
            "bg_chance_orig",
            "bg_shortcut_clean",
            "bg_anydet_clean",
            "bg_chance_clean",
            "crop_chance_majority",
        ]:
            print(f"  {k:22s} {f(b[k])} {ci(b, k) if k in b['ci95_cluster'] else ''}")
        if b["ci95_cluster"]:
            print(f"  bg_excess_clean CI {ci(b, 'bg_excess_clean')}")
    print(f"\norig hits explained by other GT organism: {out['orig_hits_explained_by_other_gt']}")
    print(f"\n{'familia':16s} medA | n_all crop bgO bgC | n_sm esp crop bgO bgC")
    for name, v in out["per_family"].items():
        a, s = v["all"], v["small"]
        if not a["n"]:
            print(f"{name:16s} n=0")
            continue
        sm = (
            f"{s['n']:4d} {s['n_specimens']:3d} {f(s['crop_acc'])} {f(s['bg_shortcut_orig'])} {f(s['bg_shortcut_clean'])}"  # noqa: E501
            if s["n"]
            else "   0"
        )
        print(
            f"{name:16s} {v['median_area']:.2f} | {a['n']:4d} {f(a['crop_acc'])} {f(a['bg_shortcut_orig'])} {f(a['bg_shortcut_clean'])} | {sm}"  # noqa: E501
        )


if __name__ == "__main__":
    main(sys.argv[1])
