"""
Cierra B6 (punto de operacion elegido en `valid`) y S1 (IC agrupados por
especimen) para un checkpoint, sin reentrenar.

Tres cosas, una sola corrida:

  (1) BARRIDO DE tau SOBRE `valid`. El umbral de operacion debe elegirse en
      validacion, nunca en test.
  (2) SIGNIFICANCIA DE tau. La curva F1 de este dataset es casi plana; elegir
      el argmax sin cuantificar su incertidumbre es sobreajustar la particion
      de validacion. Se hace un bootstrap PAREADO (misma replica para ambos
      umbrales) de F1(tau*) - F1(0.30).
  (3) BOOTSTRAP AGRUPADO SOBRE `test`. El test tiene 370 imagenes pero solo 87
      grupos independientes (rafagas del mismo individuo). Se remuestrean
      GRUPOS; el bootstrap por imagen se calcula solo para cuantificar cuanto
      subestima.

Detalles que importan para que los numeros sean los mismos que los reportados:

  - PROTOCOLOS SEPARADOS. El mAP se mide con conf=0.001 / NMS=0.7 (COCO), y el
    punto de operacion con NMS=0.6 (IOU_THRESHOLD del pipeline, config.py:56).
    Son dos pasadas distintas: mezclarlas seria reportar P/R de un NMS y mAP de
    otro. El script verifica contra model.val() que su mAP coincide.
  - EL EMPAREJAMIENTO SE RECALCULA POR CADA tau. Ultralytics empareja por IoU
    descendente, no por confianza, asi que la asignacion NO es anidada: una caja
    de baja confianza puede quedarse con el GT que le corresponderia a una de
    alta. Si se emparejara una sola vez sobre el conjunto sin filtrar y despues
    se filtrara por tau, el recall caeria artificialmente.

Uso:
    python tools/audit_sweep_and_bootstrap.py checkpoints/yolo26s_clean_best.pt \
        --data datasets/clean --out reports/paper --device cpu
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

IOUV = np.linspace(0.5, 0.95, 10)   # umbrales COCO para mAP
MATCH_IOU = 0.5                     # IoU de emparejamiento del punto de operacion
SCAN_CONF = 0.001                   # piso de confianza de las pasadas
MAP_NMS_IOU = 0.7                   # NMS del protocolo COCO
OP_NMS_IOU = 0.6                    # NMS del pipeline (config.py:56)
TAUS = np.round(np.arange(0.05, 0.9501, 0.05), 2)
BASELINE_TAU = 0.30                 # umbral actual del pipeline (config.py:53-55)


def load_gt(labels_dir: Path, stem: str, w: int = 640, h: int = 640):
    cls, box = [], []
    for ln in (labels_dir / f"{stem}.txt").read_text().splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = ln.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        cls.append(int(c))
        box.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.array(cls, int), np.array(box, np.float32).reshape(-1, 4)


def _greedy(iou: np.ndarray, thr: float, n_pred: int) -> np.ndarray:
    """Emparejamiento 1-a-1 por IoU descendente, igual que
    ultralytics.utils.metrics.ConfusionMatrix.match_predictions."""
    hit = np.zeros(n_pred, bool)
    gi, pi = np.nonzero(iou >= thr)
    if not len(gi):
        return hit
    order = np.argsort(-iou[gi, pi])
    seen_g, seen_p = set(), set()
    for g, p in zip(gi[order], pi[order], strict=True):
        if g in seen_g or p in seen_p:
            continue
        seen_g.add(g)
        seen_p.add(p)
        hit[p] = True
    return hit


def _iou_masked(pred_box, pred_cls, gt_box, gt_cls):
    if not len(gt_box) or not len(pred_box):
        return np.zeros((len(gt_box), len(pred_box)), np.float32)
    iou = box_iou(torch.from_numpy(gt_box), torch.from_numpy(pred_box)).numpy()
    return iou * (gt_cls[:, None] == pred_cls[None, :])


def predict(model, ip: Path, device: str, nms_iou: float):
    r = model.predict(str(ip), conf=SCAN_CONF, iou=nms_iou, device=device, verbose=False)[0]
    if len(r.boxes):
        return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy().astype(np.float32))
    return np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32)


def scan_map(model, ds: Path, split: str, group_of: dict, device: str, limit: int = 0):
    """Pasada para mAP: tp (n_pred, 10) sobre el conjunto completo, como Ultralytics."""
    img_dir, lbl_dir = ds / split / "images", ds / split / "labels"
    paths = sorted(img_dir.iterdir())
    if limit:
        paths = paths[:limit]
    out = []
    for n, ip in enumerate(paths, 1):
        gt_cls, gt_box = load_gt(lbl_dir, ip.stem)
        pb, pc, pf = predict(model, ip, device, MAP_NMS_IOU)
        iou = _iou_masked(pb, pc, gt_box, gt_cls)
        tp = np.zeros((len(pb), len(IOUV)), bool)
        for k, thr in enumerate(IOUV):
            tp[:, k] = _greedy(iou, thr, len(pb))
        out.append({"stem": ip.stem, "group": group_of[ip.stem], "tp": tp,
                    "conf": pf, "pred_cls": pc, "target_cls": gt_cls})
        if n % 50 == 0 or n == len(paths):
            print(f"  map/{split}: {n}/{len(paths)}", flush=True)
    return out


def scan_op(model, ds: Path, split: str, group_of: dict, device: str, nc: int,
            limit: int = 0):
    """Pasada para el punto de operacion: cuentas por cada tau de la grilla,
    con el emparejamiento RECALCULADO despues de filtrar (ver docstring)."""
    img_dir, lbl_dir = ds / split / "images", ds / split / "labels"
    paths = sorted(img_dir.iterdir())
    if limit:
        paths = paths[:limit]
    out = []
    for n, ip in enumerate(paths, 1):
        gt_cls, gt_box = load_gt(lbl_dir, ip.stem)
        pb, pc, pf = predict(model, ip, device, OP_NMS_IOU)
        tp = np.zeros((len(TAUS), nc), int)
        npred = np.zeros((len(TAUS), nc), int)
        for k, t in enumerate(TAUS):
            m = pf >= t
            hit = _greedy(_iou_masked(pb[m], pc[m], gt_box, gt_cls), MATCH_IOU, int(m.sum()))
            for c, ok in zip(pc[m], hit, strict=True):
                npred[k, c] += 1
                if ok:
                    tp[k, c] += 1
        ngt = np.zeros(nc, int)
        for c in gt_cls:
            ngt[c] += 1
        out.append({"stem": ip.stem, "group": group_of[ip.stem],
                    "tp": tp, "npred": npred, "ngt": ngt})
        if n % 50 == 0 or n == len(paths):
            print(f"  op/{split}: {n}/{len(paths)}", flush=True)
    return out


def prf_curves(items):
    """Devuelve (micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1), cada
    uno de longitud len(TAUS)."""
    tp = np.sum([it["tp"] for it in items], axis=0)          # (n_tau, nc)
    npred = np.sum([it["npred"] for it in items], axis=0)
    ngt = np.sum([it["ngt"] for it in items], axis=0)        # (nc,)

    mtp, mnp, mng = tp.sum(1), npred.sum(1), int(ngt.sum())
    mp = np.divide(mtp, mnp, out=np.zeros(len(TAUS)), where=mnp > 0)
    mr = mtp / mng if mng else np.zeros(len(TAUS))
    mf = np.divide(2 * mp * mr, mp + mr, out=np.zeros(len(TAUS)), where=(mp + mr) > 0)

    present = ngt > 0
    p_c = np.divide(tp, npred, out=np.zeros_like(tp, float), where=npred > 0)[:, present]
    r_c = np.divide(tp[:, present], ngt[present], out=np.zeros_like(tp[:, present], float),
                    where=ngt[present] > 0)
    f_c = np.divide(2 * p_c * r_c, p_c + r_c, out=np.zeros_like(p_c), where=(p_c + r_c) > 0)
    return mp, mr, mf, p_c.mean(1), r_c.mean(1), f_c.mean(1)


def maps(items):
    """mAP50 y mAP50-95 macro con la misma funcion que usa model.val()."""
    tp = np.concatenate([it["tp"] for it in items])
    conf = np.concatenate([it["conf"] for it in items])
    pcls = np.concatenate([it["pred_cls"] for it in items])
    tcls = np.concatenate([it["target_cls"] for it in items])
    if not len(tcls) or not len(tp):
        return float("nan"), float("nan")
    ap = ap_per_class(tp, conf, pcls, tcls, plot=False)[5]
    return float(ap[:, 0].mean()), float(ap.mean())


def group_index(items):
    d: dict[int, list] = {}
    for it in items:
        d.setdefault(it["group"], []).append(it)
    return d


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("checkpoint", type=Path)
    ap_.add_argument("--data", type=Path, default=Path("datasets/clean"))
    ap_.add_argument("--out", type=Path, default=Path("reports/paper"))
    ap_.add_argument("--device", default="cpu")
    ap_.add_argument("--resamples", type=int, default=1000)
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--limit", type=int, default=0,
                     help="solo N imagenes por split (prueba de humo)")
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = yaml.safe_load((args.data / "data.yaml").read_text())["names"]
    nc = len(names)
    group_of = json.loads((args.data / "split_report.json").read_text())["group_of_image"]
    model = YOLO(str(args.checkpoint))
    tag = args.checkpoint.stem
    k_base = int(np.argmin(np.abs(TAUS - BASELINE_TAU)))

    # ---------- (0) control contra model.val() ----------
    print(f"[0/4] control: model.val() sobre test (conf={SCAN_CONF}, NMS={MAP_NMS_IOU})")
    ref = model.val(data=str(args.data / "data.yaml"), split="test", device=args.device,
                    conf=SCAN_CONF, iou=MAP_NMS_IOU, plots=False, verbose=False,
                    project=str(args.out / "val_runs"), name=f"{tag}_ref", exist_ok=True)
    ref50, ref95 = float(ref.box.map50), float(ref.box.map)
    print(f"  referencia: mAP50={ref50:.4f}  mAP50-95={ref95:.4f}")

    # ---------- (1) barrido sobre valid ----------
    print(f"\n[1/4] pasada sobre valid para el punto de operacion (NMS={OP_NMS_IOU})")
    val_op = scan_op(model, args.data, "valid", group_of, args.device, nc, args.limit)
    mp, mr, mf, Mp, Mr, Mf = prf_curves(val_op)
    k_star = int(np.argmax(mf))

    sweep = [{"tau": float(TAUS[k]),
              "precision_micro": round(float(mp[k]), 4),
              "recall_micro": round(float(mr[k]), 4),
              "f1_micro": round(float(mf[k]), 4),
              "precision_macro": round(float(Mp[k]), 4),
              "recall_macro": round(float(Mr[k]), 4),
              "f1_macro": round(float(Mf[k]), 4)} for k in range(len(TAUS))]

    print(f"\n{'tau':>6}{'P':>9}{'R':>9}{'F1':>9}   (micro, IoU={MATCH_IOU:.2f})")
    for k, row in enumerate(sweep):
        mark = "  <-- max F1" if k == k_star else ("  <-- pipeline" if k == k_base else "")
        print(f"{row['tau']:>6.2f}{row['precision_micro']:>9.4f}"
              f"{row['recall_micro']:>9.4f}{row['f1_micro']:>9.4f}{mark}")

    # ---------- (2) significancia de tau ----------
    val_groups = group_index(val_op)
    vg = sorted(val_groups)
    rng = np.random.default_rng(args.seed)
    diffs, argmaxes, curves = [], [], []
    for _ in range(args.resamples):
        idx = rng.integers(0, len(vg), len(vg))
        sample = [it for i in idx for it in val_groups[vg[i]]]
        f1b = prf_curves(sample)[2]
        curves.append(f1b)
        argmaxes.append(float(TAUS[int(np.argmax(f1b))]))
        diffs.append(float(f1b[k_star] - f1b[k_base]))
    diffs, curves = np.array(diffs), np.array(curves)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    contains_zero = bool(lo <= 0 <= hi)

    for k, row in enumerate(sweep):
        cl, ch = np.percentile(curves[:, k], [2.5, 97.5])
        row["f1_micro_ci_low"] = round(float(cl), 4)
        row["f1_micro_ci_high"] = round(float(ch), 4)

    with open(args.out / "threshold_sweep_valid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)

    print(f"\n[2/4] significancia (bootstrap pareado sobre {len(vg)} grupos de valid)")
    print(f"  tau*={TAUS[k_star]:.2f} F1={mf[k_star]:.4f} | "
          f"pipeline tau={BASELINE_TAU:.2f} F1={mf[k_base]:.4f}")
    print(f"  diferencia F1(tau*)-F1(0.30) = {diffs.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print("  -> " + ("EL IC CONTIENE 0: no hay evidencia para mover el umbral."
                     if contains_zero else "el IC no contiene 0: la mejora es consistente."))
    cnt = Counter(argmaxes)
    print("  estabilidad del argmax:", ", ".join(
        f"{t:.2f}={100*c/args.resamples:.0f}%" for t, c in cnt.most_common(5)))

    tau = float(TAUS[k_star])

    # ---------- (3) test: puntual + bootstrap ----------
    print(f"\n[3/4] pasada sobre test para mAP (NMS={MAP_NMS_IOU})")
    test_map = scan_map(model, args.data, "test", group_of, args.device, args.limit)
    print(f"\n[4/4] pasada sobre test para el punto de operacion (NMS={OP_NMS_IOU})")
    test_op = scan_op(model, args.data, "test", group_of, args.device, nc, args.limit)

    m50, m95 = maps(test_map)
    d50, d95 = abs(m50 - ref50), abs(m95 - ref95)
    print(f"\nCONTROL mAP50    manual={m50:.4f} val()={ref50:.4f} delta={d50:.4f}")
    print(f"CONTROL mAP50-95 manual={m95:.4f} val()={ref95:.4f} delta={d95:.4f}")
    ok = max(d50, d95) <= 0.005
    print("  OK: mismo estimador." if ok else
          "  [ADVERTENCIA] el emparejamiento manual NO reproduce model.val().")

    c = prf_curves(test_op)
    point = {"map50": m50, "map50_95": m95,
             "precision_micro": float(c[0][k_star]), "recall_micro": float(c[1][k_star]),
             "f1_micro": float(c[2][k_star]), "precision_macro": float(c[3][k_star]),
             "recall_macro": float(c[4][k_star]), "f1_macro": float(c[5][k_star])}
    print("\npuntual (test):", json.dumps({k: round(v, 4) for k, v in point.items()},
                                          ensure_ascii=False))

    map_by_g, op_by_g = group_index(test_map), group_index(test_op)
    groups = sorted(map_by_g)
    map_by_s = {it["stem"]: it for it in test_map}
    op_by_s = {it["stem"]: it for it in test_op}
    stems = [it["stem"] for it in test_map]

    def replicate(units, dmap, dop, label):
        r = np.random.default_rng(args.seed)
        acc = {k: [] for k in point}
        for b in range(args.resamples):
            idx = r.integers(0, len(units), len(units))
            a50, a95 = maps([x for i in idx for x in dmap(units[i])])
            cc = prf_curves([x for i in idx for x in dop(units[i])])
            vals = (a50, a95, cc[0][k_star], cc[1][k_star], cc[2][k_star],
                    cc[3][k_star], cc[4][k_star], cc[5][k_star])
            for k, v in zip(point, vals, strict=True):
                acc[k].append(float(v))
            if (b + 1) % 250 == 0:
                print(f"    {label} {b+1}/{args.resamples}", flush=True)
        return {k: np.array(v, float) for k, v in acc.items()}

    print(f"\nbootstrap AGRUPADO: {args.resamples} replicas sobre {len(groups)} grupos")
    boot_g = replicate(groups, lambda g: map_by_g[g], lambda g: op_by_g[g], "grupo")
    print(f"bootstrap POR IMAGEN (contraste): {len(stems)} imagenes")
    boot_i = replicate(stems, lambda t: [map_by_s[t]], lambda t: [op_by_s[t]], "imagen")

    result = {
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "n_images": len(test_map),
        "n_groups": len(groups),
        "n_instances": int(sum(it["ngt"].sum() for it in test_op)),
        "control_vs_ultralytics_val": {
            "map50_ultralytics": round(ref50, 4), "map50_manual": round(m50, 4),
            "map50_95_ultralytics": round(ref95, 4), "map50_95_manual": round(m95, 4),
            "max_abs_delta": round(float(max(d50, d95)), 4), "reproduces": ok,
        },
        "map_protocol": {"conf_floor": SCAN_CONF, "nms_iou": MAP_NMS_IOU,
                         "iou_thresholds": "0.50:0.05:0.95"},
        "operating_point": {
            "tau": tau, "selected_on": "valid", "nms_iou": OP_NMS_IOU,
            "matching_iou": MATCH_IOU,
            "baseline_tau": BASELINE_TAU,
            "f1_valid_tau_star": round(float(mf[k_star]), 4),
            "f1_valid_baseline": round(float(mf[k_base]), 4),
            "paired_diff_mean": round(float(diffs.mean()), 4),
            "paired_diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "ci_contains_zero": contains_zero,
            "argmax_stability": {f"{t:.2f}": round(c_ / args.resamples, 4)
                                 for t, c_ in sorted(cnt.items())},
        },
        "bootstrap": {"resamples": args.resamples, "seed": args.seed,
                      "unit": "group_id (especimen)"},
        "metrics": {},
    }

    print(f"\n{'metrica':<18}{'valor':>9}   {'IC95 AGRUPADO':>20}{'ancho':>8}   "
          f"{'IC95 por imagen':>20}{'ancho':>8}  factor")
    for k in point:
        gl, gh = np.nanpercentile(boot_g[k], [2.5, 97.5])
        il, ih = np.nanpercentile(boot_i[k], [2.5, 97.5])
        wg, wi = float(gh - gl), float(ih - il)
        result["metrics"][k] = {
            "point_estimate": round(point[k], 4),
            "ci95_grouped": [round(float(gl), 4), round(float(gh), 4)],
            "ci95_grouped_width": round(wg, 4),
            "ci95_per_image": [round(float(il), 4), round(float(ih), 4)],
            "ci95_per_image_width": round(wi, 4),
            "underestimation_factor": round(wg / wi, 2) if wi > 0 else None,
            "boot_std_grouped": round(float(np.nanstd(boot_g[k])), 4),
        }
        fac = f"x{wg/wi:.2f}" if wi > 0 else "-"
        print(f"{k:<18}{point[k]:>9.4f}   [{gl:.4f}, {gh:.4f}]{wg:>8.4f}   "
              f"[{il:.4f}, {ih:.4f}]{wi:>8.4f}  {fac}")

    facs = [v["underestimation_factor"] for v in result["metrics"].values()
            if v["underestimation_factor"]]
    result["median_underestimation_factor"] = round(float(np.median(facs)), 2)
    print(f"\nEl IC por imagen es {result['median_underestimation_factor']}x el agrupado "
          "(mediana sobre las 8 metricas): ese es el factor de subestimacion.")

    (args.out / "bootstrap_grouped.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nescrito: {args.out/'threshold_sweep_valid.csv'}")
    print(f"escrito: {args.out/'bootstrap_grouped.json'}")


if __name__ == "__main__":
    main()
