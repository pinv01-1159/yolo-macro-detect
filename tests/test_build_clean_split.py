"""Chequeo del invariante que justifica todo el módulo: ningún grupo se reparte."""


import numpy as np

from tools.build_clean_split import assign, group, source_id

RATIOS = {"train": 0.7, "valid": 0.15, "test": 0.15}


def _items(sids):
    return [{"sid": s, "classes": [0]} for s in sids]


def test_source_id_quita_sufijo_roboflow():
    assert source_id("2025_0519_101711_479_JPG.rf.58833d0f.jpg") == "2025_0519_101711_479_JPG"


def test_rafaga_temporal_queda_en_un_solo_grupo():
    # tres fotos con 5s de diferencia, y una cuarta 10 minutos después
    sids = ["2025_0519_101700_1", "2025_0519_101705_2", "2025_0519_101710_3", "2025_0519_102800_4"]
    items = _items(sids)
    sim = np.eye(4, dtype=np.float32)  # sin parecido visual: solo manda el tiempo
    labels = group(items, sim, gap_s=60, vis_thr=0.95)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_similitud_visual_une_aunque_esten_lejos_en_el_tiempo():
    items = _items(["2025_0519_101700_1", "2025_1202_235900_2"])
    sim = np.array([[1.0, 0.99], [0.99, 1.0]], np.float32)
    assert len(set(group(items, sim, gap_s=60, vis_thr=0.95))) == 1


def test_cada_grupo_va_a_un_unico_split():
    items = _items([f"2025_0519_10{i:02d}00_{i}" for i in range(20)])
    groups = {g: [i for i in range(20) if i % 4 == g] for g in range(4)}
    placement, _, _ = assign(groups, items, nc=1, ratios=RATIOS, seed=0)
    assert set(placement) == set(groups)
    assert all(isinstance(v, str) for v in placement.values())
    # y las tres particiones se usan cuando hay grupos de sobra
    assert len(set(placement.values())) >= 2


def test_assign_respeta_aproximadamente_las_proporciones():
    items = _items([f"2025_0519_10{i:02d}00_{i}" for i in range(100)])
    groups = {g: [g] for g in range(100)}
    _, have, total = assign(groups, items, nc=1, ratios=RATIOS, seed=0)
    assert abs(have["train"][0] / total[0] - 0.7) < 0.05
