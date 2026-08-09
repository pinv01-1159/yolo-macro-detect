"""
Reconstruye los splits de un dataset YOLO agrupando por espécimen/ráfaga.

Problema que resuelve: las fotos se toman en ráfagas del mismo individuo
(nombre = AAAA_MMDD_HHMMSS_seq). Roboflow asigna el split por imagen, al
azar, así que cuadros casi idénticos del mismo bicho caen en train y en
valid/test a la vez. El modelo memoriza el cuadro, no la familia.

Solución: agrupar (union-find) por cercanía temporal + similitud visual, y
asignar GRUPOS enteros a un solo split, estratificando por clase.

Uso:
    python tools/build_clean_split.py datasets/v9 datasets/clean
"""

import argparse
import collections
import datetime
import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SPLITS = ("train", "valid", "test")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+(\.\w+)$")
STAMP = re.compile(r"(\d{4})_(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def source_id(name: str) -> str:
    """Nombre original de la foto, sin el sufijo .rf.<hash> de Roboflow."""
    return RF_SUFFIX.sub("", name)


def timestamp(sid: str) -> datetime.datetime | None:
    m = STAMP.match(sid)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    return datetime.datetime(y, mo, d, h, mi, s)


def scan(root: Path) -> list[dict]:
    """Una entrada por imagen fuente (descarta copias aumentadas de Roboflow)."""
    seen: set[str] = set()
    items = []
    for split in SPLITS:
        img_dir = root / split / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            sid = source_id(img.name)
            if sid in seen:
                continue
            label = root / split / "labels" / (img.stem + ".txt")
            classes = []
            if label.exists():
                classes = [int(ln.split()[0]) for ln in label.read_text().split("\n") if ln.strip()]
            if not classes:
                continue  # ponytail: imagen sin cajas = sin señal, se descarta
            seen.add(sid)
            items.append(
                {"sid": sid, "orig_split": split, "img": img, "label": label, "classes": classes}
            )
    return items


def thumbnails(items: list[dict], size: int = 32) -> np.ndarray:
    """Vectores L2-normalizados de miniaturas en gris, normalizadas por contraste."""
    feats = np.zeros((len(items), size * size), np.float32)
    for i, it in enumerate(items):
        a = np.asarray(
            Image.open(it["img"]).convert("L").resize((size, size), Image.BILINEAR), np.float32
        )
        feats[i] = ((a - a.mean()) / (a.std() + 1e-6)).ravel()
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def group(items: list[dict], sim: np.ndarray, gap_s: float, vis_thr: float) -> list[int]:
    """Union-find: une fotos consecutivas en el tiempo o visualmente casi idénticas."""
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    times = [timestamp(it["sid"]) for it in items]
    order = [i for i in np.argsort([t.timestamp() if t else 0 for t in times]) if times[i]]
    for a, b in zip(order, order[1:], strict=False):
        if (times[b] - times[a]).total_seconds() <= gap_s:
            union(int(a), int(b))

    ii, jj = np.where(np.triu(sim, 1) > vis_thr)
    for a, b in zip(ii, jj, strict=True):
        union(int(a), int(b))

    return [find(i) for i in range(len(items))]


def assign(
    groups: dict[int, list[int]],
    items: list[dict],
    nc: int,
    ratios: dict[str, float],
    seed: int,
):
    """Greedy estratificado: cada grupo entero va al split con mayor déficit de sus clases."""
    total = np.zeros(nc)
    gclasses = {}
    for gid, idxs in groups.items():
        v = np.zeros(nc)
        for i in idxs:
            for c in items[i]["classes"]:
                v[c] += 1
        gclasses[gid] = v
        total += v

    quota = {s: total * r for s, r in ratios.items()}
    have = {s: np.zeros(nc) for s in ratios}
    out: dict[int, str] = {}

    rng = np.random.default_rng(seed)
    order = sorted(groups, key=lambda g: (-gclasses[g].sum(), rng.random()))
    for gid in order:
        v = gclasses[gid]
        # déficit relativo: cuánto le falta a cada split de las clases de este grupo
        def deficit(s: str, v: np.ndarray = v) -> float:
            return float(((quota[s] - have[s]) * v).sum() / (quota[s].sum() + 1e-9))

        best = max(ratios, key=deficit)
        out[gid] = best
        have[best] += v
    return out, have, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--gap", type=float, default=60.0, help="segundos entre fotos de una ráfaga")
    ap.add_argument("--vis", type=float, default=0.95, help="coseno mínimo para unir visualmente")
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = yaml.safe_load((args.src / "data.yaml").read_text())
    names = meta["names"]
    items = scan(args.src)
    raw = sum(1 for _ in (args.src / "train" / "images").iterdir())
    print(f"{len(items)} imágenes fuente únicas (train original tenía {raw} con augmentación)")

    sim = thumbnails(items)
    sim = sim @ sim.T
    labels = group(items, sim, args.gap, args.vis)
    groups = collections.defaultdict(list)
    for i, g in enumerate(labels):
        groups[g].append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"{len(groups)} grupos (ráfagas). tamaño medio {np.mean(sizes):.1f}, máx {sizes[0]}")

    orig_leak = sum(
        len(v) for v in groups.values() if len({items[i]["orig_split"] for i in v}) > 1
    )
    print(f"LEAKAGE ORIGINAL: {orig_leak}/{len(items)} imágenes ({orig_leak/len(items):.1%}) "
          f"en grupos repartidos entre splits")

    r = [float(x) for x in args.ratios.split(",")]
    ratios = dict(zip(SPLITS, r, strict=True))
    placement, have, total = assign(groups, items, len(names), ratios, args.seed)

    # escribir dataset
    if args.dst.exists():
        shutil.rmtree(args.dst)
    for s in SPLITS:
        (args.dst / s / "images").mkdir(parents=True)
        (args.dst / s / "labels").mkdir(parents=True)
    split_of = {}
    for gid, idxs in groups.items():
        for i in idxs:
            s = placement[gid]
            split_of[i] = s
            it = items[i]
            shutil.copy2(it["img"], args.dst / s / "images" / f"{it['sid']}.jpg")
            shutil.copy2(it["label"], args.dst / s / "labels" / f"{it['sid']}.txt")

    (args.dst / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(args.dst.resolve()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )

    # verificación: ¿queda algún casi-duplicado cruzando splits?
    arr = np.array([split_of[i] for i in range(len(items))])
    report = {"groups": len(groups), "images": len(items), "leak_before": orig_leak, "residual": {}}
    tr = np.where(arr == "train")[0]
    for s in ("valid", "test"):
        idx = np.where(arr == s)[0]
        best = sim[np.ix_(idx, tr)].max(1)
        res = {f">{t:.2f}": float((best > t).mean()) for t in (0.99, 0.95, 0.90)}
        res["median"] = float(np.median(best))
        report["residual"][s] = res
        print(f"residual {s}: mediana coseno al vecino de train {res['median']:.3f} | "
              f">0.99 {res['>0.99']:.1%}  >0.95 {res['>0.95']:.1%}  >0.90 {res['>0.90']:.1%}")

    counts = {s: collections.Counter() for s in SPLITS}
    for i, s in split_of.items():
        for c in items[i]["classes"]:
            counts[s][names[c]] += 1
    report["images_per_split"] = {s: int((arr == s).sum()) for s in SPLITS}
    report["instances_per_split"] = {s: dict(counts[s]) for s in SPLITS}
    report["groups_per_split"] = collections.Counter(placement.values())
    # trazabilidad: qué ráfaga generó cada imagen, para reproducir el split
    report["group_of_image"] = {items[i]["sid"]: int(g) for i, g in enumerate(labels)}
    (args.dst / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\nimágenes:", report["images_per_split"])
    print(f"{'clase':<18}{'train':>7}{'valid':>7}{'test':>7}")
    for n in names:
        print(f"{n:<18}{counts['train'][n]:>7}{counts['valid'][n]:>7}{counts['test'][n]:>7}")


if __name__ == "__main__":
    main()
