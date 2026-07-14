"""
Modelos para el proyecto YOLO Macroinvertebrados.

Este paquete contiene las clases y módulos para entrenamiento
e inferencia de modelos YOLO.
"""

from .inference import YOLOInference
from .trainer import YOLOTrainer

__all__ = ['YOLOTrainer', 'YOLOInference']
