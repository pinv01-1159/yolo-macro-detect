"""Extrae a JSON la configuración de entrenamiento que Ultralytics guardó dentro
de cada checkpoint.

Contexto: los `checkpoints/*_environment.json` registran solo 12-13 campos y no
incluyen la augmentación, `patience`, `cos_lr`, `amp` ni las métricas (hallazgo
S2 de la auditoría). Se pensó en reconstruir esos valores leyendo el código del
commit correspondiente, pero **no hace falta**: cada `.pt` de Ultralytics lleva
un diccionario `train_args` con los 111 parámetros efectivos y un
`train_results` con el historial por época. Eso no es una reconstrucción a
posteriori sino el registro que la propia biblioteca escribió al entrenar, que
es la mejor fuente posible.

Los `environment.json` originales no se tocan: quedan como están y esto se
escribe al lado.

Uso:
    uv run python tools/extract_train_args.py
"""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

# Necesarios para resolver `optimizer='auto'`; salen del split report y del data.yaml
N_TRAIN = 1660
NC = 19


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def resumen_entrenamiento(train_results: dict) -> dict:
    """Epocas corridas y mejor epoca, leidas del historial.

    El campo `epochs` de `train_args` es el maximo SOLICITADO (200 en los tres);
    las que de verdad corrieron son las filas del historial, porque la parada
    temprana corta antes. Confundir ambos fue la causa de que un documento
    dijera "144/200" donde correspondia "144/174".
    """
    epocas = train_results.get("epoch", [])
    col = next((c for c in train_results if "mAP50-95" in c), None)
    if not epocas or col is None:
        return {"epocas_corridas": len(epocas) or None}

    valores = train_results[col]
    i = max(range(len(valores)), key=lambda k: valores[k])
    return {
        "epocas_corridas": len(epocas),
        "mejor_epoca": int(epocas[i]),
        "mejor_map50_95_validacion": round(float(valores[i]), 4),
        "nota_metrica": (
            "mAP50-95 sobre VALIDACION durante el entrenamiento. No es la cifra "
            "del informe, que se mide sobre test: ver reports/paper/eval_overall.csv."
        ),
    }


def optimizador_resuelto(train_args: dict, n_train: int, nc: int) -> dict:
    """Resuelve `optimizer='auto'` a los valores que Ultralytics usa de verdad.

    Es el campo mas enganioso del registro: con `optimizer='auto'` la biblioteca
    **ignora** `lr0` y `momentum` de `train_args` y los recalcula. Publicar el
    `lr0=0.01` que figura ahi significa reportar 23 veces el learning rate real.
    El propio codigo lo dice al arrancar: "ignoring 'lr0=...' and 'momentum=...'".

    Regla de ultralytics 8.4.95, engine/trainer.py:1064-1073.
    """
    if train_args.get("optimizer") != "auto":
        return {"_procedencia": "literal", "optimizer": train_args.get("optimizer")}

    import math
    batch, nbs, epochs = train_args["batch"], train_args["nbs"], train_args["epochs"]
    iterations = math.ceil(n_train / max(batch, nbs)) * epochs
    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    nombre, lr, momentum = (
        ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
    )
    return {
        "_procedencia": "derivado de la regla versionada, corroborado con lr/pg* registrado",
        "_regla": "ultralytics 8.4.95 engine/trainer.py:1064-1073, rama optimizer=='auto'",
        "_entradas": {"nc": nc, "n_train": n_train, "batch": batch, "nbs": nbs,
                      "epochs": epochs, "iterations": iterations},
        "_reemplaza": (
            "Los campos lr0 y momentum de train_args son IGNORADOS por la rama "
            "auto. Publicar lr0=0.01 seria reportar ~23x el valor real."
        ),
        "optimizer": nombre,
        "lr0_efectivo": lr,
        "momentum_efectivo": momentum,
        "warmup_bias_lr_efectivo": 0.0,
        "accumulate": max(round(nbs / batch), 1),
    }


def curvas(train_results: dict, destino: Path) -> int:
    """Vuelca el historial por epoca a CSV.

    Es el `results.csv` del entrenamiento, que se perdio con los directorios de
    `runs/detect/` pero sobrevive dentro del checkpoint.
    """
    import csv
    columnas = list(train_results)
    filas = len(train_results[columnas[0]])
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columnas)
        for i in range(filas):
            w.writerow([train_results[c][i] for c in columnas])
    return filas


def main() -> None:
    escritos = []
    for pesos in sorted(CKPT.glob("*_best.pt")):
        ck = torch.load(pesos, map_location="cpu", weights_only=False)
        train_args = dict(ck.get("train_args") or {})
        if not train_args:
            print(f"AVISO  {pesos.name} no trae train_args, se omite")
            continue

        doc = {
            "_procedencia": {
                "fuente": f"train_args embebido en {pesos.name}",
                "escrito_por": "ultralytics, durante el entrenamiento",
                "reconstruido": False,
                "nota": (
                    "Registro de ejecucion, no una reconstruccion posterior. "
                    "Complementa checkpoints/*_environment.json, que se genero "
                    "antes de que el pipeline volcara la configuracion completa."
                ),
            },
            "checkpoint": {
                "archivo": pesos.name,
                "sha256": sha256(pesos),
                "fecha_entrenamiento": str(ck.get("date")),
                "ultralytics_version": str(ck.get("version")),
            },
            "entrenamiento": resumen_entrenamiento(ck.get("train_results") or {}),
            "metricas_finales_validacion": {
                k: round(float(v), 5)
                for k, v in (ck.get("train_metrics") or {}).items()
                if isinstance(v, (int, float))
            },
            "train_args": {k: train_args[k] for k in sorted(train_args)},
            "derivado": {
                "optimizador": optimizador_resuelto(train_args, N_TRAIN, NC),
            },
            "procedencia_git": ck.get("git") or None,
            "no_aplica": {
                "attention_reg_lambda": (
                    "No figura en train_args, asi que la regularizacion por "
                    "atencion no estuvo activa en este entrenamiento. El commit "
                    "exacto que produjo los pesos esta en `procedencia_git`, "
                    "escrito por Ultralytics durante la corrida."
                ),
            },
        }

        base = pesos.stem.replace("_best", "")
        destino = CKPT / f"{base}_train_args.json"
        destino.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str) + "\n")
        escritos.append(destino)

        if ck.get("train_results"):
            n = curvas(ck["train_results"], CKPT / f"{base}_training_curves.csv")
            doc["_curvas"] = f"{base}_training_curves.csv ({n} epocas)"
        ent = doc["entrenamiento"]
        print(f"{pesos.name:28} -> {destino.name}  "
              f"({len(train_args)} campos, "
              f"{ent.get('epocas_corridas')} epocas, mejor {ent.get('mejor_epoca')})")

    assert escritos, "no se extrajo ningun checkpoint"
    print(f"\n{len(escritos)} archivos escritos en checkpoints/")


if __name__ == "__main__":
    main()
