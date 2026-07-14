"""
Calculadora de Índice BMWP para Evaluación de Calidad del Agua
============================================================

Este módulo implementa el cálculo del índice BMWP (Biological Monitoring Working Party)
basado en las familias de macroinvertebrados detectadas por el sistema YOLO.

El índice BMWP es un método ampliamente utilizado para evaluar la calidad
ecológica del agua en cuerpos de agua dulce.

Autor: Kevin Galeano
Proyecto: PINV01-1159
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class BMWPResult:
    """Resultado del cálculo del índice BMWP."""
    total_score: int
    water_quality_class: str
    water_quality_description: str
    family_scores: list[dict[str, Any]]
    confidence: float


class BMWPCalculator:
    """
    Calculadora del índice BMWP para evaluación de calidad del agua.

    Esta clase implementa el cálculo del índice BMWP basado en las familias
    de macroinvertebrados detectadas, siguiendo los criterios establecidos
    para Paraguay y adaptados de Roldán (2003) y Alba-Tercedor (1996).
    """

    # Puntajes BMWP por familia (basado en el paper)
    FAMILY_SCORES = {
        "Belostomatidae": 5,
        "Coenagrionidae": 7,
        "Dytiscidae": 3,
        "Physidae": 3,
        "Planorbidae": 5,
        "Chironomidae": 8,
        "Noteridae": 4,
        "Libellulidae": 8,
        "Hirudinidae": 9
    }

    # Clasificación de calidad del agua según BMWP
    WATER_QUALITY_CLASSES = {
        "I": {"min": 101, "max": float('inf'), "description": "Muy limpia"},
        "II": {"min": 61, "max": 100, "description": "Aceptable"},
        "III": {"min": 36, "max": 60, "description": "Dudosa"},
        "IV": {"min": 16, "max": 35, "description": "Crítica"},
        "V": {"min": 0, "max": 15, "description": "Muy crítica"}
    }

    def __init__(self):
        """Inicializa la calculadora BMWP."""
        self.available_families = set(self.FAMILY_SCORES.keys())

    def calculate_bmwp(self, detections: list[dict[str, Any]]) -> BMWPResult:
        """
        Calcula el índice BMWP basado en las detecciones de macroinvertebrados.

        Args:
            detections: Lista de detecciones con formato:
                [
                    {
                        "familia": "Physidae",
                        "cantidad": 6,
                        "confidence_promedio": 0.95
                    },
                    ...
                ]

        Returns:
            BMWPResult con el puntaje total, clase de calidad y detalles
        """
        total_score = 0
        family_scores = []
        total_confidence = 0.0
        total_detections = 0

        # Procesar cada detección
        for detection in detections:
            family = detection.get("familia", "")
            quantity = detection.get("cantidad", 0)
            confidence = detection.get("confidence_promedio", 0.0)

            # Verificar si la familia está en nuestro mapeo
            if family in self.FAMILY_SCORES:
                family_score = self.FAMILY_SCORES[family]
                contribution = family_score * quantity
                total_score += contribution

                family_scores.append({
                    "familia": family,
                    "cantidad": quantity,
                    "bmwp_individual": family_score,
                    "bmwp_contribution": contribution,
                    "confidence": confidence
                })

                # Acumular confianza ponderada
                total_confidence += confidence * quantity
                total_detections += quantity

        # Calcular confianza promedio ponderada
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0

        # Determinar clase de calidad del agua
        water_quality_class, water_quality_description = self._get_water_quality_class(total_score)

        return BMWPResult(
            total_score=total_score,
            water_quality_class=water_quality_class,
            water_quality_description=water_quality_description,
            family_scores=family_scores,
            confidence=round(avg_confidence, 3)
        )

    def _get_water_quality_class(self, total_score: int) -> tuple[str, str]:
        """
        Determina la clase de calidad del agua basada en el puntaje BMWP.

        Args:
            total_score: Puntaje total BMWP

        Returns:
            Tupla con (clase, descripción)
        """
        for class_name, criteria in self.WATER_QUALITY_CLASSES.items():
            if criteria["min"] <= total_score <= criteria["max"]:
                return class_name, criteria["description"]

        # Fallback para casos extremos
        return "V", "Muy crítica"

    def get_family_score(self, family: str) -> int | None:
        """
        Obtiene el puntaje BMWP para una familia específica.

        Args:
            family: Nombre de la familia

        Returns:
            Puntaje BMWP o None si no está disponible
        """
        return self.FAMILY_SCORES.get(family)

    def get_available_families(self) -> list[str]:
        """
        Obtiene la lista de familias disponibles para el cálculo BMWP.

        Returns:
            Lista de familias disponibles
        """
        return list(self.available_families)

    def get_water_quality_info(self) -> dict[str, dict[str, Any]]:
        """
        Obtiene información sobre las clases de calidad del agua.

        Returns:
            Diccionario con información de las clases
        """
        return self.WATER_QUALITY_CLASSES.copy()

    def format_result_for_json(self, bmwp_result: BMWPResult) -> dict[str, Any]:
        """
        Formatea el resultado BMWP para exportación JSON.

        Args:
            bmwp_result: Resultado del cálculo BMWP

        Returns:
            Diccionario formateado para JSON
        """
        return {
            "bmwp_total": bmwp_result.total_score,
            "calidad_agua": (
                f"{bmwp_result.water_quality_description} "
                f"(Clase {bmwp_result.water_quality_class})"
            ),
            "clase_calidad": bmwp_result.water_quality_class,
            "descripcion_calidad": bmwp_result.water_quality_description,
            "confianza": bmwp_result.confidence,
            "detalles_familias": [
                {
                    "familia": score["familia"],
                    "cantidad": score["cantidad"],
                    "bmwp": score["bmwp_individual"],
                    "contribucion": score["bmwp_contribution"],
                    "confianza": score["confidence"]
                }
                for score in bmwp_result.family_scores
            ]
        }

    def validate_detections(self, detections: list[dict[str, Any]]) -> list[str]:
        """
        Valida las detecciones y retorna familias no reconocidas.

        Args:
            detections: Lista de detecciones

        Returns:
            Lista de familias no reconocidas
        """
        unrecognized = []
        for detection in detections:
            family = detection.get("familia", "")
            if family and family not in self.FAMILY_SCORES:
                unrecognized.append(family)

        return unrecognized


# Instancia global para uso fácil
bmwp_calculator = BMWPCalculator()
