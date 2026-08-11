"""
Regularización por atención (estilo "Right for the Right Reasons", Ross
et al. 2017) para penalizar directamente el uso del fondo, en vez de
esperar que la augmentación de datos lo resuelva sola.

Motivación (ver docs/leakage_analysis.md sección 3 y tools/background_ablation.py):
la augmentación copy-paste (tools/copy_paste_augment.py) no bajó la tasa de
atajo por fondo ni con 79% de imágenes sintéticas en train -- la hipótesis
verificada es que el modelo aprendió a distinguir imágenes "pegadas" (con
costura visible: borde recto, salto de grano/iluminación) de fotos reales,
y solo aplica "ignorá el fondo" al primer caso, algo que ni las fotos reales
de entrenamiento ni de evaluación disparan. Diversificar los datos no fuerza
al modelo a soltar el atajo si puede separar limpiamente el régimen sintético
del real.

Esta regularización ataca el mecanismo directamente en vez de esperar que
aparezca de los datos: penaliza la magnitud de activación en preds["feats"]
(los mapas de features multiescala que Ultralytics expone justo antes de la
cabeza de detección -- compartidos por cualquier arquitectura YOLO, por eso
generaliza sin tocar nada arquitectura-específico) en las celdas de la
grilla que caen FUERA de cualquier caja GT, sea la imagen real o sintética.

No es una implementación literal de RRR (que penaliza el gradiente de la
pérdida respecto al PIXEL de entrada, con backward doble -- caro). Es una
versión más simple y barata: penaliza directamente la ACTIVACIÓN de una
representación interna compartida en las regiones de fondo, en un solo
backward pass. Mismo espíritu ("no mires el fondo"), mucho menor costo.
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


def _parse_output(preds: Any) -> dict[str, torch.Tensor]:
    """Misma lógica que v8DetectionLoss.parse_output: preds puede venir como
    tupla (raw, dict) o directamente como dict, según el modo del forward."""
    return preds[1] if isinstance(preds, tuple) else preds


def background_activation_penalty(
    feats: list[torch.Tensor], batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Penaliza la magnitud de activación (media de cuadrados sobre canales)
    en las celdas de la grilla de cada escala que caen fuera de toda caja GT.

    Args:
        feats: lista de tensores [B, C, H_s, W_s], uno por escala de detección.
        batch: batch de Ultralytics con 'batch_idx' y 'bboxes' (cx,cy,w,h
            normalizados [0,1] respecto de la imagen ya preprocesada -- mismo
            sistema de coordenadas que cubren feats).

    Returns:
        Escalar diferenciable: promedio (sobre escalas) de la razón
        activación-de-fondo / activación-de-objeto. Normalizado (no magnitud
        cruda) para que el peso lambda_bg tenga una escala interpretable y
        estable durante todo el entrenamiento, independiente de cómo cambie
        la magnitud absoluta de las activaciones época a época.
    """
    batch_idx = batch["batch_idx"]
    bboxes = batch["bboxes"]
    device = feats[0].device
    batch_size = feats[0].shape[0]
    eps = 1e-6

    scale_penalties = []
    for feat in feats:
        _, _, h, w = feat.shape
        activation = feat.pow(2).mean(dim=1)  # (B, H, W): magnitud por celda

        fg_mask = torch.zeros(batch_size, h, w, device=device, dtype=torch.bool)
        for i in range(batch_size):
            sel = batch_idx == i
            if not sel.any():
                continue
            cx, cy, bw, bh = bboxes[sel].unbind(-1)
            x1 = ((cx - bw / 2) * w).clamp(0, w).long()
            y1 = ((cy - bh / 2) * h).clamp(0, h).long()
            x2 = ((cx + bw / 2) * w).ceil().clamp(0, w).long()
            y2 = ((cy + bh / 2) * h).ceil().clamp(0, h).long()
            for x1i, y1i, x2i, y2i in zip(
                x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(), strict=False
            ):
                if x2i > x1i and y2i > y1i:
                    fg_mask[i, y1i:y2i, x1i:x2i] = True

        bg_mask = ~fg_mask
        n_bg = bg_mask.sum().clamp(min=1)
        n_fg = fg_mask.sum().clamp(min=1)
        bg_activation = (activation * bg_mask).sum() / n_bg
        fg_activation = (activation * fg_mask).sum() / n_fg
        scale_penalties.append(bg_activation / (fg_activation + eps))

    return torch.stack(scale_penalties).mean()


class AttentionPenaltyLoss:
    """Envuelve la pérdida base (v8DetectionLoss, o E2ELoss para arquitecturas
    end2end como yolo26 -- misma lógica que usa DetectionModel.init_criterion)
    y le suma la penalización de activación de fondo. Misma interfaz (preds,
    batch) -> (loss_total, loss_items) que espera BaseModel.loss(), así que
    se puede asignar directo a `model.criterion`."""

    def __init__(self, model: torch.nn.Module, lambda_bg: float = 0.05):
        self.is_e2e = bool(getattr(model, "end2end", False))
        self.base = E2ELoss(model) if self.is_e2e else v8DetectionLoss(model)
        self.lambda_bg = lambda_bg

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, loss_items = self.base(preds, batch)

        parsed = _parse_output(preds)
        if self.is_e2e:
            # dos cabezas (one2many/one2one) comparten el mismo backbone --
            # penalizar la union de ambas listas de feats para que la
            # regularizacion alcance a la que efectivamente se usa en inferencia
            feats = [*parsed["one2many"]["feats"], *parsed["one2one"]["feats"]]
        else:
            feats = parsed["feats"]

        penalty = background_activation_penalty(feats, batch)
        weighted_penalty = self.lambda_bg * penalty.to(loss.dtype)

        loss = loss + weighted_penalty
        loss_items = torch.cat([loss_items, weighted_penalty.detach().unsqueeze(0)])
        return loss, loss_items

    def update(self) -> None:
        """Delegate al decay one2many/one2one de E2ELoss si corresponde
        (BaseTrainer llama esto cada época via hasattr(criterion, "update"))."""
        if hasattr(self.base, "update"):
            self.base.update()


class _InitCriterion:
    """Reemplazo picklable de DetectionModel.init_criterion como atributo de
    instancia. Una lambda/función local rompe torch.save() (que hace
    deepcopy/pickle del modelo al guardar checkpoints) porque pickle no
    puede resolver el nombre calificado de una función anidada -- por eso
    esto es una clase definida a nivel de módulo en vez de un closure."""

    def __init__(self, model: torch.nn.Module, lambda_bg: float):
        self.model = model
        self.lambda_bg = lambda_bg

    def __call__(self) -> AttentionPenaltyLoss:
        return AttentionPenaltyLoss(self.model, lambda_bg=self.lambda_bg)


class AttentionRegularizedTrainer(DetectionTrainer):
    """DetectionTrainer que instala AttentionPenaltyLoss como criterio del
    modelo, y agrega "bg_loss" a las columnas de progreso/results.csv para
    poder ver la penalización bajar durante el entrenamiento."""

    lambda_bg: float = 0.05

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        # No se puede construir AttentionPenaltyLoss (envuelve v8DetectionLoss,
        # que lee model.args) aca todavia -- el trainer recien asigna
        # model.args DESPUES de que get_model() retorna. Hay que replicar la
        # misma pereza que usa BaseModel.loss() (self.criterion =
        # self.init_criterion(), solo en el primer forward), parcheando
        # init_criterion en vez de setear criterion directo.
        model.init_criterion = _InitCriterion(model, self.lambda_bg)
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (*self.loss_names, "bg_loss")
        return validator


def make_attention_regularized_trainer(lambda_bg: float) -> type[AttentionRegularizedTrainer]:
    """model.train(trainer=...) instancia la clase que se le pasa sin
    argumentos extra propios -- no hay forma de inyectar lambda_bg vía
    constructor. Se crea una subclase con el valor grabado como atributo de
    clase en su lugar."""

    class _ConfiguredTrainer(AttentionRegularizedTrainer):
        pass

    _ConfiguredTrainer.lambda_bg = lambda_bg
    return _ConfiguredTrainer
