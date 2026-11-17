"""Reproduccion puntual + bootstrap agrupado por especimen (y por imagen) sobre los
stats por imagen capturados de DetectionValidator (capture_stats.py). Solo numpy +
ultralytics.utils.metrics.ap_per_class (la misma funcion que usa model.val())."""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import ap_per_class

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reports/paper"
TAGS = ["yolo11s", "yolo12s", "yolo26s"]
B, SEED, TOL = 1000, 42, 0.002
METRICS = ["map50", "map50_95", "precision", "recall", "f1"]

group_of = json.loads((ROOT / "datasets/clean/split_report.json").read_text())["group_of_image"]
ref_csv = {
    r["model"]: r
    for r in csv.DictReader(open(ROOT / "reports/paper/eval_overall.csv"))
    if r["protocol"] == "coco"
}


def metrics(items):
    tp = np.concatenate([i["tp"] for i in items])
    conf = np.concatenate([i["conf"] for i in items])
    pc = np.concatenate([i["pred_cls"] for i in items])
    tc = np.concatenate([i["target_cls"] for i in items])
    _, _, p, r, _, ap, *_ = ap_per_class(tp, conf, pc, tc, plot=False)
    P, R = float(p.mean()), float(r.mean())
    return {
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
        "precision": P,
        "recall": R,
        "f1": 2 * P * R / (P + R) if P + R else 0.0,
    }


data = {t: pickle.load(open(HERE / "bootstrap_stats" / f"stats_{t}.pkl", "rb")) for t in TAGS}
names = [i["im_name"] for i in data[TAGS[0]]["images"]]
for t in TAGS:
    assert [i["im_name"] for i in data[t]["images"]] == names
stems = [Path(n).stem for n in names]
gid = np.array([group_of[s] for s in stems])
groups = np.unique(gid)
members = [np.nonzero(gid == g)[0] for g in groups]
print(f"{len(stems)} imagenes, {len(groups)} grupos")

# ---- 1) reproduccion puntual ----
repro, point, ok_all = {}, {}, True
for t in TAGS:
    imgs = data[t]["images"]
    point[t] = metrics(imgs)
    ref = ref_csv[t]
    rows = {}
    for k in METRICS:
        rv = float(ref[k])
        d = abs(point[t][k] - rv)
        rows[k] = {
            "ours": round(point[t][k], 4),
            "eval_overall_csv": round(rv, 4),
            "abs_diff": float(f"{d:.2e}"),
            "ok": d < TOL,
        }
        ok_all &= d < TOL
    rows["val_rerun_this_session"] = {k: round(v, 4) for k, v in data[t]["val_ref"].items()}
    repro[t] = rows
    print(t, {k: (rows[k]["ours"], rows[k]["eval_overall_csv"], rows[k]["ok"]) for k in METRICS})
if not ok_all:
    (HERE / "bootstrap_ultralytics_exact.json").write_text(
        json.dumps({"reproduction": repro}, indent=2)
    )
    raise SystemExit("NO REPRODUCE: se detiene antes de los IC")


# ---- 2) bootstrap (mismas replicas para los tres modelos -> pareado) ----
def run(units_fn, n_units, label):
    rng = np.random.default_rng(SEED)
    acc = {t: {k: np.empty(B) for k in METRICS} for t in TAGS}
    for b in range(B):
        idx = units_fn(rng.integers(0, n_units, n_units))
        for t in TAGS:
            imgs = data[t]["images"]
            m = metrics([imgs[i] for i in idx])
            for k in METRICS:
                acc[t][k][b] = m[k]
        if (b + 1) % 200 == 0:
            print(f"  {label} {b + 1}/{B}", flush=True)
    return acc


boot_g = run(lambda gi: np.concatenate([members[g] for g in gi]), len(groups), "agrupado")
boot_i = run(lambda ii: ii, len(stems), "por imagen")

ci = lambda a: [round(float(x), 4) for x in np.percentile(a, [2.5, 97.5])]  # noqa: E731
out_models = {}
for t in TAGS:
    out_models[t] = {
        k: {
            "point": round(point[t][k], 4),
            "ci95_grouped": ci(boot_g[t][k]),
            "ci95_per_image": ci(boot_i[t][k]),
            "width_ratio_grouped_over_image": round(
                float(
                    np.diff(np.percentile(boot_g[t][k], [2.5, 97.5]))[0]
                    / max(np.diff(np.percentile(boot_i[t][k], [2.5, 97.5]))[0], 1e-12)
                ),
                2,
            ),
            "boot_std_grouped": round(float(boot_g[t][k].std()), 4),
        }
        for k in METRICS
    }

pairs = {}
for a, b in [("yolo12s", "yolo11s"), ("yolo26s", "yolo11s"), ("yolo12s", "yolo26s")]:
    pairs[f"{a}-{b}"] = {}
    for k in ["map50_95", "map50"]:
        dg = boot_g[a][k] - boot_g[b][k]
        di = boot_i[a][k] - boot_i[b][k]
        lo, hi = np.percentile(dg, [2.5, 97.5])
        pairs[f"{a}-{b}"][k] = {
            "point_diff": round(point[a][k] - point[b][k], 4),
            "ci95_grouped": [round(float(lo), 4), round(float(hi), 4)],
            "ci95_per_image": ci(di),
            "significant_grouped": bool(lo > 0 or hi < 0),
            "frac_reps_positive_grouped": round(float((dg > 0).mean()), 3),
        }
    print(a, "-", b, pairs[f"{a}-{b}"]["map50_95"])

res = {
    "protocol": {
        "split": "test",
        "conf": 0.001,
        "nms_iou": 0.7,
        "iou_thresholds": "0.50:0.05:0.95",
        "matching": "DetectionValidator.match_predictions (stats capturados de model.val())",
        "ap": "ultralytics.utils.metrics.ap_per_class",
        "precision_recall": "macro (media por clase) en el umbral de conf de max F1 medio, como box.mp/box.mr",  # noqa: E501
        "f1": "2PR/(P+R) sobre P,R macro (igual que eval_overall.csv)",
    },
    "bootstrap": {
        "resamples": B,
        "seed": SEED,
        "unit_grouped": "group_id (especimen), split_report.json",
        "n_images": len(stems),
        "n_groups": int(len(groups)),
        "paired": "mismas replicas (indices) para los tres modelos",
        "ci": "percentil 2.5/97.5",
    },
    "reproduction_vs_eval_overall_csv": repro,
    "models": out_models,
    "paired_differences": pairs,
}
(HERE / "bootstrap_ultralytics_exact.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False)
)
print(json.dumps(out_models, indent=1))
