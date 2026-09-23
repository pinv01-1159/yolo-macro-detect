"""
Calculadora del Índice BMWP para Evaluación de Calidad del Agua
===============================================================

Implementa el índice BMWP (Biological Monitoring Working Party) en su
adaptación neotropical **BMWP/Col** (Roldán Pérez, 2003), a partir de las
familias de macroinvertebrados detectadas por el sistema de visión.

Cómo funciona el índice
-----------------------
BMWP es un índice de **presencia/ausencia**, no de abundancia. Cada familia
encontrada en un sitio aporta su puntaje **una sola vez**, sin importar cuántos
ejemplares se hayan hallado. Encontrar cuarenta quironómidos no indica mejor
calidad de agua que encontrar uno: indica lo mismo, la presencia de un taxón
tolerante a la contaminación.

El índice se define **por sitio de muestreo**, agregando todas las familias
encontradas allí. Calcularlo sobre una fotografía individual no tiene sentido
ecológico: una foto contiene típicamente un solo taxón, y el índice de un solo
taxón no es un índice. Por eso la API expone `calculate_site()`, que recibe la
recolección completa de un sitio, y `aggregate_images()`, que combina las
detecciones de varias fotografías en un único registro de sitio.

Junto al BMWP se reporta el **ASPT** (Average Score Per Taxon), el promedio de
puntaje por familia. El ASPT es menos sensible al esfuerzo de muestreo: un
sitio donde se buscó más tiempo acumula más familias y sube su BMWP aunque la
calidad sea la misma, mientras que el ASPT se mantiene estable.

Autor: Kevin Galeano
Proyecto: PINV01-1159

Referencias
-----------
- Roldán Pérez, G. (2003). *Bioindicación de la calidad del agua en Colombia:
  uso del método BMWP/Col.* Universidad de Antioquia, Medellín.
- Alba-Tercedor, J. & Sánchez-Ortega, A. (1988). Un método rápido y simple para
  evaluar la calidad biológica de las aguas corrientes basado en el de
  Hellawell (1978). *Limnetica*, 4, 51-56.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

# Procedencia de cada puntaje, para que el origen del número sea auditable.
# "bmwp_col": figura en la tabla BMWP/Col de Roldán (2003).
# "proximidad": la familia no figura en la tabla; el puntaje se asigna por
#               proximidad taxonómica y ecológica, y está marcado como
#               provisional hasta que un especialista lo confirme.
Procedencia = Literal["bmwp_col", "proximidad"]


class PuntajeFamilia(TypedDict):
    """Puntaje BMWP de una familia, con el origen del valor."""
    score: int
    procedencia: Procedencia
    nota: str


class WaterQualityClassInfo(TypedDict):
    """Estructura de una entrada de clasificación de calidad del agua."""
    min: float
    max: float
    description: str


@dataclass
class BMWPResult:
    """Resultado del cálculo del índice BMWP para un sitio de muestreo."""
    total_score: int
    water_quality_class: str
    water_quality_description: str
    family_scores: list[dict[str, Any]]
    confidence: float
    aspt: float = 0.0
    n_families_scored: int = 0
    unscored_families: list[str] = field(default_factory=list)
    provisional_families: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BMWPCalculator:
    """
    Calculadora del índice BMWP/Col para evaluación de calidad del agua.

    El índice es de presencia/ausencia y se define por sitio de muestreo.
    Ver el docstring del módulo para el fundamento.
    """

    # Puntajes BMWP/Col (Roldán, 2003) para las 19 familias que el sistema
    # reconoce. Un puntaje alto indica un taxón sensible a la contaminación;
    # uno bajo, un taxón tolerante.
    FAMILY_SCORES: dict[str, PuntajeFamilia] = {
        # --- Coleoptera ---
        "Dytiscidae": {
            "score": 9, "procedencia": "bmwp_col",
            "nota": "Coleóptero depredador de aguas bien oxigenadas.",
        },
        "Noteridae": {
            "score": 7, "procedencia": "bmwp_col",
            "nota": "Coleóptero de aguas someras con vegetación.",
        },
        "Hydrophilidae": {
            "score": 3, "procedencia": "bmwp_col",
            "nota": "Tolerante; frecuente en aguas con materia orgánica.",
        },
        # --- Hemiptera ---
        "Gerridae": {
            "score": 8, "procedencia": "bmwp_col",
            "nota": "Patinador de superficie; asociado a aguas limpias.",
        },
        "Notonectidae": {
            "score": 7, "procedencia": "bmwp_col",
            "nota": "Nadador de espalda, aguas lénticas de calidad media-alta.",
        },
        "Belostomatidae": {
            "score": 5, "procedencia": "bmwp_col",
            "nota": "Chinche de agua gigante; amplio rango de tolerancia.",
        },
        "Miridae": {
            "score": 5, "procedencia": "proximidad",
            "nota": (
                "No figura en la tabla BMWP/Col: la familia es mayormente "
                "terrestre y sus representantes acuáticos son marginales. "
                "Se asigna por analogía con otros Hemiptera de tolerancia "
                "intermedia. PROVISIONAL: requiere confirmación de un "
                "especialista antes de usarse en un reporte oficial."
            ),
        },
        # --- Odonata ---
        "Coenagrionidae": {
            "score": 7, "procedencia": "bmwp_col",
            "nota": "Caballito del diablo; calidad media-alta.",
        },
        "Libellulidae": {
            "score": 6, "procedencia": "bmwp_col",
            "nota": "Libélula; tolerancia intermedia.",
        },
        # --- Diptera ---
        "Ceratopogonidae": {
            "score": 3, "procedencia": "bmwp_col",
            "nota": "Larva tolerante, sedimentos finos.",
        },
        "Psychodidae": {
            "score": 3, "procedencia": "bmwp_col",
            "nota": "Tolerante; aguas con alta carga orgánica.",
        },
        "Chironomidae": {
            "score": 2, "procedencia": "bmwp_col",
            "nota": (
                "Muy tolerante a la contaminación orgánica y a la baja "
                "concentración de oxígeno. Su dominancia es un indicador "
                "clásico de deterioro."
            ),
        },
        # --- Mollusca: Gastropoda ---
        "Ampullariidae": {
            "score": 6, "procedencia": "bmwp_col",
            "nota": "Caracol manzana; tolerancia intermedia.",
        },
        "Ancylidae": {
            "score": 6, "procedencia": "bmwp_col",
            "nota": "Lapa de agua dulce, sustratos duros oxigenados.",
        },
        "Planorbidae": {
            "score": 5, "procedencia": "bmwp_col",
            "nota": "Caracol planórbido; aguas lénticas con vegetación.",
        },
        "Physidae": {
            "score": 3, "procedencia": "bmwp_col",
            "nota": "Tolerante; frecuente en aguas eutrofizadas.",
        },
        # --- Mollusca: Bivalvia ---
        "Hyriidae": {
            "score": 6, "procedencia": "proximidad",
            "nota": (
                "Bivalvo unionoideo sudamericano, ausente de la tabla "
                "BMWP/Col original. Se asigna por analogía con otros bivalvos "
                "filtradores de vida larga, que requieren sustrato estable y "
                "agua de calidad sostenida. PROVISIONAL: requiere confirmación "
                "de un especialista antes de usarse en un reporte oficial."
            ),
        },
        # --- Hirudinea (sanguijuelas) ---
        "Glossiphoniidae": {
            "score": 3, "procedencia": "bmwp_col",
            "nota": "Sanguijuela tolerante a la contaminación orgánica.",
        },
        "Hirudinidae": {
            "score": 3, "procedencia": "bmwp_col",
            "nota": (
                "Sanguijuela tolerante. Un puntaje alto para este taxón "
                "invertiría el sentido ecológico del índice."
            ),
        },
    }

    # Clasificación de calidad del agua según BMWP/Col (Roldán, 2003)
    WATER_QUALITY_CLASSES: dict[str, WaterQualityClassInfo] = {
        "I": {"min": 101, "max": float("inf"), "description": "Muy limpia"},
        "II": {"min": 61, "max": 100, "description": "Aceptable"},
        "III": {"min": 36, "max": 60, "description": "Dudosa"},
        "IV": {"min": 16, "max": 35, "description": "Crítica"},
        "V": {"min": 1, "max": 15, "description": "Muy crítica"},
    }

    def __init__(self) -> None:
        self.available_families = set(self.FAMILY_SCORES.keys())

    # ------------------------------------------------------------------
    # Agregación por sitio
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_images(per_image_detections: list[list[dict[str, Any]]]
                         ) -> list[dict[str, Any]]:
        """Combina las detecciones de varias fotografías en un registro de sitio.

        El BMWP se define por sitio de muestreo, así que antes de calcularlo hay
        que reunir todo lo encontrado allí. Las cantidades se suman ---son
        informativas, aunque no entren en el índice--- y la confianza se
        promedia ponderada por cantidad.

        Args:
            per_image_detections: una lista de detecciones por cada fotografía.

        Returns:
            Lista de familias del sitio, apta para `calculate_site()`.
        """
        acumulado: dict[str, dict[str, float]] = {}
        for detections in per_image_detections:
            for det in detections or []:
                familia = det.get("familia", "")
                if not familia:
                    continue
                cantidad = float(det.get("cantidad", 0) or 0)
                conf = float(det.get("confidence_promedio", 0.0) or 0.0)
                acc = acumulado.setdefault(familia, {"cantidad": 0.0, "conf_pond": 0.0})
                acc["cantidad"] += cantidad
                acc["conf_pond"] += conf * cantidad

        return [
            {
                "familia": familia,
                "cantidad": int(acc["cantidad"]),
                "confidence_promedio": (
                    acc["conf_pond"] / acc["cantidad"] if acc["cantidad"] else 0.0
                ),
            }
            for familia, acc in sorted(acumulado.items())
        ]

    # ------------------------------------------------------------------
    # Cálculo
    # ------------------------------------------------------------------

    def calculate_site(self, detections: list[dict[str, Any]]) -> BMWPResult:
        """Calcula el BMWP de un **sitio de muestreo**.

        Cada familia presente aporta su puntaje una sola vez, con independencia
        de cuántos ejemplares se hayan detectado.

        Args:
            detections: familias encontradas en el sitio, con formato
                ``{"familia": str, "cantidad": int, "confidence_promedio": float}``.
                Usar `aggregate_images()` si se parte de varias fotografías.

        Returns:
            BMWPResult con puntaje, ASPT, clase de calidad y trazabilidad de
            qué familias entraron y cuáles no.
        """
        presentes: dict[str, dict[str, Any]] = {}
        for det in detections or []:
            familia = det.get("familia", "")
            if not familia:
                continue
            acc = presentes.setdefault(
                familia, {"cantidad": 0, "conf_pond": 0.0}
            )
            cantidad = int(det.get("cantidad", 0) or 0)
            acc["cantidad"] += cantidad
            acc["conf_pond"] += float(det.get("confidence_promedio", 0.0) or 0.0) * cantidad

        total_score = 0
        family_scores: list[dict[str, Any]] = []
        unscored: list[str] = []
        provisional: list[str] = []
        conf_pond_total = 0.0
        cantidad_total = 0

        for familia in sorted(presentes):
            acc = presentes[familia]
            cantidad = int(acc["cantidad"])
            conf = acc["conf_pond"] / cantidad if cantidad else 0.0
            cantidad_total += cantidad
            conf_pond_total += acc["conf_pond"]

            entrada = self.FAMILY_SCORES.get(familia)
            if entrada is None:
                # No se descarta en silencio: queda registrado en el resultado.
                unscored.append(familia)
                continue

            # Presencia/ausencia: el puntaje entra una vez, no por ejemplar.
            total_score += entrada["score"]
            if entrada["procedencia"] == "proximidad":
                provisional.append(familia)

            family_scores.append({
                "familia": familia,
                "cantidad": cantidad,
                "bmwp_individual": entrada["score"],
                "procedencia": entrada["procedencia"],
                "nota": entrada["nota"],
                "confidence": round(conf, 3),
            })

        n_scored = len(family_scores)
        aspt = total_score / n_scored if n_scored else 0.0
        avg_confidence = conf_pond_total / cantidad_total if cantidad_total else 0.0

        warnings: list[str] = []
        if n_scored == 0:
            warnings.append(
                "No se identificó ninguna familia con puntaje BMWP: el índice no "
                "es calculable. Ausencia de datos no equivale a mala calidad."
            )
        elif n_scored < 3:
            warnings.append(
                f"Solo {n_scored} familia(s) con puntaje. El BMWP de un sitio "
                f"con tan pocos taxones es poco informativo; considerar el ASPT "
                f"({aspt:.1f}) y ampliar el esfuerzo de muestreo."
            )
        if unscored:
            warnings.append(
                "Familias detectadas sin puntaje en la tabla, excluidas del "
                f"índice: {', '.join(unscored)}."
            )
        if provisional:
            warnings.append(
                "Puntaje provisional por proximidad taxonómica (requiere "
                f"validación de un especialista): {', '.join(provisional)}."
            )

        clase, descripcion = self._get_water_quality_class(total_score, n_scored)

        return BMWPResult(
            total_score=total_score,
            water_quality_class=clase,
            water_quality_description=descripcion,
            family_scores=family_scores,
            confidence=round(avg_confidence, 3),
            aspt=round(aspt, 2),
            n_families_scored=n_scored,
            unscored_families=unscored,
            provisional_families=provisional,
            warnings=warnings,
        )

    def calculate_bmwp(self, detections: list[dict[str, Any]]) -> BMWPResult:
        """Alias de `calculate_site()`, por compatibilidad con la API previa."""
        return self.calculate_site(detections)

    def _get_water_quality_class(self, total_score: int, n_scored: int) -> tuple[str, str]:
        """Clase de calidad a partir del puntaje BMWP.

        Sin familias puntuadas no hay índice: devolver "Muy crítica" ante la
        ausencia de detecciones confundiría "no medimos" con "el agua está mal",
        que es justo el error que un sistema de bioindicación no puede cometer.
        """
        if n_scored == 0:
            return "N/D", "No determinable (sin familias con puntaje)"

        for class_name, criteria in self.WATER_QUALITY_CLASSES.items():
            if criteria["min"] <= total_score <= criteria["max"]:
                return class_name, criteria["description"]
        return "V", "Muy crítica"

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get_family_score(self, family: str) -> int | None:
        """Puntaje BMWP de una familia, o None si no está en la tabla."""
        entrada = self.FAMILY_SCORES.get(family)
        return entrada["score"] if entrada else None

    def get_family_entry(self, family: str) -> PuntajeFamilia | None:
        """Entrada completa (puntaje, procedencia y nota) de una familia."""
        return self.FAMILY_SCORES.get(family)

    def get_available_families(self) -> list[str]:
        """Familias con puntaje BMWP disponible."""
        return sorted(self.available_families)

    def validate_detections(self, detections: list[dict[str, Any]]) -> list[str]:
        """Familias detectadas que no tienen puntaje en la tabla.

        Se devuelven para que quien llame pueda avisarlo: excluirlas del índice
        es correcto según el método, pero hacerlo en silencio no.
        """
        return sorted({
            det.get("familia", "")
            for det in detections or []
            if det.get("familia") and det["familia"] not in self.FAMILY_SCORES
        })

    def get_water_quality_info(self) -> dict[str, WaterQualityClassInfo]:
        """Clases de calidad del agua y sus rangos."""
        return self.WATER_QUALITY_CLASSES.copy()

    def format_result_for_json(self, bmwp_result: BMWPResult) -> dict[str, Any]:
        """Formatea el resultado para exportación JSON."""
        return {
            "bmwp_total": bmwp_result.total_score,
            "aspt": bmwp_result.aspt,
            "n_familias_puntuadas": bmwp_result.n_families_scored,
            "clase_calidad": bmwp_result.water_quality_class,
            "descripcion_calidad": bmwp_result.water_quality_description,
            # composicion legible, para mostrar de una sola pieza
            "calidad_agua": (
                f"{bmwp_result.water_quality_description} "
                f"(Clase {bmwp_result.water_quality_class})"
            ),
            "familias": bmwp_result.family_scores,
            "familias_sin_puntaje": bmwp_result.unscored_families,
            "familias_provisionales": bmwp_result.provisional_families,
            "confianza_promedio": bmwp_result.confidence,
            "advertencias": bmwp_result.warnings,
            "metodo": "BMWP/Col (Roldán, 2003) — presencia/ausencia, por sitio",
        }


bmwp_calculator = BMWPCalculator()
