"""
Módulo de inferencia para modelos YOLO de macroinvertebrados.

Este módulo maneja la inferencia y predicción de macroinvertebrados
usando modelos YOLO entrenados, incluyendo evaluación de calidad del agua
mediante el índice BMWP.
"""

import base64
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import config
from utils.bmwp_calculator import BMWPResult, bmwp_calculator
from utils.logger import get_inference_logger
from utils.validators import (
    validate_confidence_threshold,
    validate_image_path,
    validate_iou_threshold,
    validate_model_path,
)


class YOLOInference:
    """
    Clase para manejar la inferencia con modelos YOLO.

    Esta clase encapsula toda la lógica de inferencia, incluyendo
    carga de modelos, predicción, anotación, cálculo BMWP y exportación de resultados.
    """

    def __init__(self, model_path: str | None = None):
        """
        Inicializa el sistema de inferencia.

        Args:
            model_path: Ruta al modelo entrenado (opcional)
        """
        self.logger = get_inference_logger()
        self.model: YOLO | None = None
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> YOLO:
        """
        Carga el modelo YOLO para inferencia.

        Args:
            model_path: Ruta al modelo entrenado

        Returns:
            Modelo YOLO cargado
        """
        self.logger.info(f"Cargando modelo para inferencia: {model_path}")

        try:
            # Validar modelo
            validate_model_path(model_path)

            # Cargar modelo
            self.model = YOLO(model_path)
            self.model_path = model_path

            self.logger.info("Modelo cargado exitosamente")
            self.logger.info(f"   - Clases: {list(self.model.names.values())}")

            return self.model

        except Exception as e:
            self.logger.error(f"Error al cargar el modelo: {e}")
            raise

    def predict_image(self,
                     image_path: str | Path,
                     conf_threshold: float | None = None,
                     iou_threshold: float | None = None,
                     img_size: int | None = None,
                     save_annotated: bool = True,
                     output_dir: str = "results",
                     calculate_bmwp: bool = False) -> dict[str, Any]:
        """
        Realiza predicción en una imagen.

        Args:
            image_path: Ruta a la imagen
            conf_threshold: Umbral de confianza (usa config por defecto)
            iou_threshold: Umbral de IoU para NMS (usa config por defecto)
            img_size: Tamaño de imagen para inferencia (usa config por defecto)
            save_annotated: Si guardar la imagen anotada
            output_dir: Directorio para guardar resultados
            calculate_bmwp: Si calcular el índice BMWP

        Returns:
            Diccionario con resultados de la predicción
        """
        # `or` trataria 0.0 como "no especificado"; conf=0.0 es un valor legitimo
        conf_threshold = (
            conf_threshold if conf_threshold is not None else config.confidence_threshold
        )
        iou_threshold = iou_threshold if iou_threshold is not None else config.iou_threshold
        img_size = img_size if img_size is not None else config.img_size

        self.logger.info(f"Realizando predicción en: {image_path}")
        self.logger.info(f"   - Umbral confianza: {conf_threshold}")
        self.logger.info(f"   - Umbral IoU: {iou_threshold}")
        self.logger.info(f"   - Tamaño imagen: {img_size}")
        self.logger.info(f"   - Cálculo BMWP: {calculate_bmwp}")

        try:
            validate_image_path(image_path)
            validate_confidence_threshold(conf_threshold)
            validate_iou_threshold(iou_threshold)

            if self.model is None:
                raise ValueError("Modelo no cargado. Use load_model() primero.")

            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")

            # conf/iou se pasan directamente al modelo: Ultralytics aplica el
            # umbral de confianza y el NMS con el IoU configurado antes de
            # devolver las detecciones, así que no hace falta re-filtrar acá.
            # ultralytics' Model.__call__ is typed to return
            # Iterator[...] | list[Results] | list[Tensor] regardless of the
            # `stream` argument's value, so mypy can't narrow indexability
            # from the signature alone. stream defaults to False here, which
            # always yields a list, so indexing is safe at runtime.
            results = self.model(
                frame, imgsz=img_size, conf=conf_threshold, iou=iou_threshold, verbose=False
            )[0]  # type: ignore[index]
            detections = sv.Detections.from_ultralytics(results)

            result_data = self._process_detections(detections, results, frame)

            if calculate_bmwp and result_data['detecciones']:
                bmwp_result = self.calculate_bmwp(result_data['detecciones'])
                parcial = bmwp_calculator.format_result_for_json(bmwp_result)
                # El BMWP se define por SITIO de muestreo, no por fotografia:
                # una foto contiene tipicamente un solo taxon y el indice de un
                # solo taxon no describe la calidad del agua. Se marca como
                # parcial para que nadie lo lea como un veredicto de sitio; el
                # indice real lo produce predict_batch() agregando el sitio.
                parcial["parcial"] = True
                parcial["alcance"] = "imagen"
                parcial["advertencias"] = [
                    "Aporte de una sola fotografia; no es el indice del sitio. "
                    "Usar predict_batch() sobre todas las fotos del sitio.",
                    *parcial.get("advertencias", []),
                ]
                result_data.update(parcial)
                self.logger.info(
                    f"Aporte BMWP de esta imagen: {bmwp_result.total_score} "
                    f"(parcial, no es el indice del sitio)"
                )

            if len(detections) > 0:
                annotated_frame = self._annotate_image(frame, detections, results)
                result_data["imagen_anotada_base64"] = self._encode_image(annotated_frame)

                if save_annotated:
                    self._save_annotated_image(annotated_frame, image_path, output_dir)
            else:
                result_data["imagen_anotada_base64"] = self._encode_image(frame)
                self.logger.info("No se detectaron macroinvertebrados")

                if calculate_bmwp:
                    # Ausencia de deteccion no es evidencia de mala calidad:
                    # es ausencia de evidencia. Declarar "Muy critica" acá
                    # inventaria un veredicto ambiental a partir de una foto
                    # vacia.
                    result_data.update({
                        "bmwp_total": 0,
                        "aspt": 0.0,
                        "n_familias_puntuadas": 0,
                        "clase_calidad": "N/D",
                        "descripcion_calidad": "No determinable (sin detecciones)",
                        "familias": [],
                        "confianza_promedio": 0.0,
                        "parcial": True,
                        "alcance": "imagen",
                        "advertencias": [
                            "Sin detecciones: el indice no es calculable. "
                            "Ausencia de datos no equivale a mala calidad."
                        ],
                    })

            self.logger.info("Predicción completada")
            self.logger.info(f"   - Total detecciones: {result_data['total_detecciones']}")

            return result_data

        except Exception as e:
            self.logger.error(f"Error durante la predicción: {e}")
            raise

    def calculate_bmwp(self, detections: list[dict[str, Any]]) -> BMWPResult:
        """
        Calcula el índice BMWP basado en las detecciones.

        Args:
            detections: Lista de detecciones de macroinvertebrados

        Returns:
            Resultado del cálculo BMWP
        """
        self.logger.info("Calculando índice BMWP...")

        try:
            # Validar detecciones no reconocidas
            unrecognized = bmwp_calculator.validate_detections(detections)
            if unrecognized:
                self.logger.warning(f"Familias no reconocidas para BMWP: {unrecognized}")

            # Calcular BMWP
            bmwp_result = bmwp_calculator.calculate_bmwp(detections)

            self.logger.info("BMWP calculado")
            self.logger.info(f"   - Puntaje: {bmwp_result.total_score}")
            self.logger.info(f"   - ASPT: {bmwp_result.aspt}")
            self.logger.info(f"   - Familias puntuadas: {bmwp_result.n_families_scored}")
            for aviso in bmwp_result.warnings:
                self.logger.warning(f"   {aviso}")

            return bmwp_result

        except Exception as e:
            self.logger.error(f"Error al calcular BMWP: {e}")
            raise

    def _process_detections(self,
                           detections: sv.Detections,
                           results,
                           frame: np.ndarray) -> dict[str, Any]:
        """
        Procesa las detecciones y genera estadísticas.

        Args:
            detections: Detecciones de supervision
            results: Resultados de YOLO
            frame: Imagen original

        Returns:
            Datos procesados de las detecciones
        """
        # Contar detecciones por familia
        family_count: defaultdict[str, int] = defaultdict(int)
        family_confidence = defaultdict(list)

        # Verificar que tenemos datos válidos
        if (detections.class_id is not None and
            detections.confidence is not None and
            len(detections.class_id) > 0):

            for class_id, conf in zip(detections.class_id, detections.confidence, strict=True):
                if class_id < len(results.names):
                    class_name = results.names[class_id]
                    family_count[class_name] += 1
                    family_confidence[class_name].append(conf)

        # Preparar resultado
        detecciones = []
        for family, count in family_count.items():
            avg_conf = np.mean(family_confidence[family])
            detecciones.append({
                "familia": family,
                "cantidad": count,
                "confidence_promedio": round(float(avg_conf), 3),
                "confidence_min": round(float(min(family_confidence[family])), 3),
                "confidence_max": round(float(max(family_confidence[family])), 3)
            })

        return {
            "detecciones": detecciones,
            "total_detecciones": len(detections),
            "familias_detectadas": len(family_count),
            "imagen_anotada_base64": None  # Se llenará después
        }

    def _annotate_image(self,
                       frame: np.ndarray,
                       detections: sv.Detections,
                       results) -> np.ndarray:
        """
        Anota la imagen con las detecciones.

        Args:
            frame: Imagen original
            detections: Detecciones
            results: Resultados de YOLO

        Returns:
            Imagen anotada
        """
        # Verificar que tenemos datos válidos para anotar
        if (detections.class_id is None or
            detections.confidence is None or
            len(detections.class_id) == 0):
            return frame

        # Crear etiquetas
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence, strict=True):
            if class_id < len(results.names):
                class_name = results.names[class_id]
                labels.append(f"{class_name}: {confidence:.2f}")

        # Configurar anotadores
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#00FF00")
        )
        label_annotator = sv.LabelAnnotator()

        # Aplicar anotaciones
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        # supervision's @ensure_cv2_image_for_class_method decorator accepts
        # both np.ndarray and PIL.Image.Image at runtime (see
        # supervision/utils/conversion.py), but is typed as `-> F`, which
        # preserves the wrapped method's narrower `Image.Image`-only
        # signature. This is a stub gap in the third-party library, not a
        # real type error: annotated_frame is always an ndarray here.
        annotated_frame = label_annotator.annotate(  # type: ignore[assignment]
            scene=annotated_frame,  # type: ignore[arg-type]
            detections=detections,
            labels=labels
        )

        return annotated_frame

    def _encode_image(self, frame: np.ndarray) -> str:
        """
        Codifica una imagen en base64.

        Args:
            frame: Imagen como array de numpy

        Returns:
            Imagen codificada en base64
        """
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def _save_annotated_image(self,
                             annotated_frame: np.ndarray,
                             original_path: str | Path,
                             output_dir: str):
        """
        Guarda la imagen anotada.

        Args:
            annotated_frame: Imagen anotada
            original_path: Ruta de la imagen original
            output_dir: Directorio de salida
        """
        try:
            # Crear directorio si no existe
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generar nombre de archivo
            original_name = Path(original_path).stem
            output_file = output_path / f"{original_name}_annotated.jpg"

            # Guardar imagen
            cv2.imwrite(str(output_file), annotated_frame)

            self.logger.info(f"Imagen anotada guardada: {output_file}")

        except Exception as e:
            self.logger.warning(f"No se pudo guardar la imagen anotada: {e}")

    def predict_batch(self,
                     image_paths: list[str | Path],
                     conf_threshold: float | None = None,
                     iou_threshold: float | None = None,
                     img_size: int | None = None,
                     save_annotated: bool = True,
                     output_dir: str = "results",
                     calculate_bmwp: bool = False) -> list[dict[str, Any]]:
        """
        Realiza predicción en un lote de imágenes.

        Args:
            image_paths: Lista de rutas de imágenes
            conf_threshold: Umbral de confianza
            iou_threshold: Umbral de IoU para NMS
            img_size: Tamaño de imagen
            save_annotated: Si guardar imágenes anotadas
            output_dir: Directorio de salida
            calculate_bmwp: Si calcular el índice BMWP del sitio

        Returns:
            Lista de resultados por imagen. Si `calculate_bmwp`, se agrega al
            final un registro con `alcance="sitio"` que contiene el índice
            agregado: es el único que describe la calidad del agua.
        """
        self.logger.info(f"Procesando lote de {len(image_paths)} imágenes")

        results = []
        for i, image_path in enumerate(image_paths, 1):
            try:
                self.logger.info(f"Procesando imagen {i}/{len(image_paths)}: {image_path}")
                # El BMWP por imagen no se calcula acá aunque se haya pedido:
                # el indice es por sitio y se agrega mas abajo. Calcularlo por
                # foto solo produciria indices parciales sin significado y una
                # advertencia por cada imagen, que con 50 fotos vuelve el log
                # ilegible justo para quien mas lo necesita leer.
                result = self.predict_image(
                    image_path=image_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    img_size=img_size,
                    save_annotated=save_annotated,
                    output_dir=output_dir,
                    calculate_bmwp=False,
                )
                result["imagen_path"] = str(image_path)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error procesando {image_path}: {e}")
                results.append({
                    "imagen_path": str(image_path),
                    "error": str(e),
                    "detecciones": [],
                    "total_detecciones": 0
                })

        if calculate_bmwp:
            # El BMWP se define por sitio de muestreo. Recién acá, con todas
            # las fotos del sitio reunidas, el índice significa algo.
            site_detections = bmwp_calculator.aggregate_images(
                [r.get("detecciones", []) for r in results if not r.get("error")]
            )
            site_result = bmwp_calculator.calculate_site(site_detections)
            site_record = bmwp_calculator.format_result_for_json(site_result)
            site_record["alcance"] = "sitio"
            site_record["parcial"] = False
            site_record["n_imagenes"] = len(results)
            results.append(site_record)

            self.logger.info(
                f"BMWP del sitio: {site_result.total_score} "
                f"(ASPT {site_result.aspt}, {site_result.n_families_scored} familias) "
                f"— {site_result.water_quality_description}"
            )
            for aviso in site_result.warnings:
                self.logger.warning(f"   {aviso}")

        self.logger.info("Procesamiento de lote completado")
        return results

    def export_results(self,
                      results: dict[str, Any] | list[dict[str, Any]],
                      output_file: str = "results/prediction_results.json"):
        """
        Exporta los resultados a un archivo JSON.

        Args:
            results: Resultados de predicción
            output_file: Archivo de salida
        """
        try:
            # Crear directorio si no existe
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Exportar resultados
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Resultados exportados a: {output_file}")

        except Exception as e:
            self.logger.error(f"Error al exportar resultados: {e}")
            raise
