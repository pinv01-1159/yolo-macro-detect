"""Genera las figuras nuevas del informe tecnico a partir de los CSV de auditoria.

Solo las dos que Ultralytics no produce: la relacion forma-desempenio y la
distribucion de tamanos de caja. Las curvas P/R/F1/PR y las matrices de
confusion se copian tal cual desde reports/paper/val_runs/yolo26s_coco/.
"""
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "reports" / "paper"
OUT = ROOT / "informe" / "imagenes"

BLUE = "#2a78d6"      # slot categorico 1 - el cuerpo de los datos
ORANGE = "#eb6834"    # slot 2 - los dos taxones filiformes que dominan el efecto
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

DESTACADAS = {"Chironomidae", "Ceratopogonidae"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def leer(nombre):
    with open(PAPER / nombre, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def elongacion_por_familia():
    """Mediana de max(w/h, h/w) por familia.

    Hay que agregar despues de aplicar max(), no antes: max() de la mediana del
    AR no es la mediana de las elongaciones y subestima a los taxones curvos,
    que aparecen tanto horizontales como verticales.
    """
    por_fam = {}
    for f in leer("bbox_instances.csv"):
        ar = float(f["aspect_ratio"])
        por_fam.setdefault(f["family"], []).append(max(ar, 1.0 / ar))
    # statistics.median promedia los dos centrales con n par; usarlo mantiene
    # la cifra alineada con reports/paper/family_size_vs_performance.csv
    return {fam: statistics.median(v) for fam, v in por_fam.items()}


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy)


def fig_forma_vs_desempenio():
    filas = leer("family_size_vs_performance.csv")
    elong = elongacion_por_familia()
    xs = [elong[f["familia"]] for f in filas]
    ys = [float(f["map50_95_mean"]) for f in filas]
    nombres = [f["familia"] for f in filas]
    r = pearson(xs, ys)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # recta de minimos cuadrados, como referencia visual de la tendencia
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    pend = cov / sum((x - mx) ** 2 for x in xs)
    x0, x1 = min(xs) - 0.10, max(xs) + 0.45
    ax.plot([x0, x1], [my + pend * (x0 - mx), my + pend * (x1 - mx)],
            color=INK2, linewidth=1.2, linestyle="--", alpha=0.55, zorder=1)

    for x, y, nom in zip(xs, ys, nombres, strict=True):
        alta = nom in DESTACADAS
        ax.scatter(x, y, s=76 if alta else 52, color=ORANGE if alta else BLUE,
                   edgecolors=SURFACE, linewidths=1.6, zorder=3)
        if alta:
            ax.annotate(f"{nom}\nmAP {y:.3f}", (x, y), textcoords="offset points",
                        xytext=(-14, 12), ha="right", va="bottom", fontsize=9,
                        color=INK, fontweight="bold", linespacing=1.3)

    # las compactas se apinan: una sola etiqueta representativa evita el choque
    ax.annotate("Las otras 17 familias caen en 0.82–0.94\n"
                "sin un predictor geométrico dominante:\n"
                "el efecto lo dominan los dos taxones filiformes",
                (0.03, 0.14), xycoords="axes fraction", ha="left", va="center",
                fontsize=8.5, color=INK2, linespacing=1.5)

    ax.set_xlabel("Elongación de la caja — mediana de máx(w/h, h/w)")
    ax.set_ylabel("mAP@0.5:0.95  (media de 3 arquitecturas)")
    ax.set_title("La forma predice el desempeño; el tamaño no", fontsize=11.5,
                 fontweight="bold", pad=16, loc="left")
    ax.text(0, 1.015, f"r = {r:.3f}   R² = {r*r:.3f}   n = 19 familias",
            transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlim(x0, x1)
    fig.tight_layout()
    fig.savefig(OUT / "elongacion_vs_map.png", dpi=200)
    plt.close(fig)
    return r


def fig_distribucion_tamanos():
    filas = leer("bbox_instances.csv")
    fracs = [float(f["area_frac"]) for f in filas]
    areas = [float(f["area_px2"]) for f in filas]
    n = len(fracs)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    mediana = sorted(fracs)[n // 2]
    ax.hist(fracs, bins=40, range=(0, 1), color=BLUE, edgecolor=SURFACE, linewidth=0.8)

    # los umbrales COCO caen fuera del eje: hay que decirlo, no dibujar una linea invisible
    umbral_large = 9216 / (640 * 640)   # 0.0225
    ax.axvline(umbral_large, color=ORANGE, linewidth=1.8, zorder=4)
    ax.annotate("umbral COCO «large» = 0.023\n(el mínimo del dataset es 0.069, 3.0× mayor)",
                (umbral_large, ax.get_ylim()[1] * 0.93), textcoords="offset points",
                xytext=(12, 0), ha="left", va="top", fontsize=9, color=ORANGE,
                fontweight="bold", linespacing=1.35)

    ax.axvline(mediana, color=INK2, linewidth=1.2, linestyle="--", zorder=4)
    ax.annotate(f"mediana = {mediana:.3f}", (mediana, ax.get_ylim()[1] * 0.78),
                textcoords="offset points", xytext=(10, 0), ha="left",
                fontsize=9, color=INK2, fontweight="bold")

    ax.set_xlabel("Área de la caja / área de la imagen  (canvas 640×640 px)")
    ax.set_ylabel("Instancias")
    ax.set_title("No hay objetos pequeños en el conjunto", fontsize=11.5,
                 fontweight="bold", pad=16, loc="left")
    ax.text(0, 1.015,
            f"las {n} instancias son «large» en escala COCO  ·  "
            f"mediana {mediana:.3f} del cuadro",
            transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "distribucion_tamanos_bbox.png", dpi=200)
    plt.close(fig)
    return min(areas), sorted(fracs)[n // 2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = fig_forma_vs_desempenio()
    area_min, frac_med = fig_distribucion_tamanos()
    # chequeo: si estas invariantes se rompen, la redaccion del informe deja de ser cierta
    assert -0.90 < r < -0.78, f"correlacion elongacion-mAP fuera de lo esperado: {r}"
    assert area_min > 9216, f"aparecio una caja no-large: {area_min} px2"
    assert 0.5 < frac_med < 0.55, f"fraccion mediana inesperada: {frac_med}"
    print(f"OK  r={r:.3f}  area_min={area_min:.0f}px2  frac_mediana={frac_med:.3f}")
    print(f"escritas: {OUT/'elongacion_vs_map.png'}")
    print(f"          {OUT/'distribucion_tamanos_bbox.png'}")


if __name__ == "__main__":
    main()
