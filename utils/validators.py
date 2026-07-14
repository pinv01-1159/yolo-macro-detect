"""
Validaciones para el proyecto YOLO Macroinvertebrados.

Este módulo contiene funciones de validación para rutas de archivos,
modelos y otros datos de entrada.
"""

import os
from pathlib import Path


def validate_image_path(image_path: str | Path) -> bool:
    """
    Valida que la ruta de imagen existe y tiene una extensión soportada.

    No decodifica la imagen: la carga real (cv2.imread) ocurre inmediatamente
    después en el código que llama a este validador, y produce un error
    igual de claro si el archivo está corrupto — decodificarla acá también
    duplicaría el trabajo.

    Args:
        image_path: Ruta a la imagen

    Returns:
        True si la ruta es válida

    Raises:
        FileNotFoundError: Si la imagen no existe
        ValueError: Si el archivo no es un archivo o la extensión no es válida
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"No es un archivo válido: {image_path}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Extensión de archivo no válida: {image_path.suffix}. "
            f"Extensiones válidas: {valid_extensions}"
        )

    return True


def validate_model_path(model_path: str | Path) -> bool:
    """
    Valida que la ruta del modelo existe y tiene la extensión correcta.

    No carga el modelo: la carga real (YOLO(...)) ocurre inmediatamente
    después en el código que llama a este validador.

    Args:
        model_path: Ruta al modelo

    Returns:
        True si la ruta es válida

    Raises:
        FileNotFoundError: Si el modelo no existe
        ValueError: Si el archivo no es un archivo o no tiene extensión .pt
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    if not model_path.is_file():
        raise ValueError(f"No es un archivo válido: {model_path}")

    if model_path.suffix.lower() != '.pt':
        raise ValueError(f"El modelo debe tener extensión .pt: {model_path}")

    return True


def validate_data_yaml_path(data_yaml_path: str | Path) -> bool:
    """
    Valida que el archivo data.yaml existe y tiene el formato correcto.

    Args:
        data_yaml_path: Ruta al archivo data.yaml

    Returns:
        True si el archivo es válido

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no tiene el formato correcto
    """
    data_yaml_path = Path(data_yaml_path)

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Archivo data.yaml no encontrado: {data_yaml_path}")

    if not data_yaml_path.is_file():
        raise ValueError(f"No es un archivo válido: {data_yaml_path}")

    if data_yaml_path.suffix.lower() != '.yaml':
        raise ValueError(f"El archivo debe tener extensión .yaml: {data_yaml_path}")

    try:
        import yaml
        with open(data_yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Clave requerida '{key}' no encontrada en {data_yaml_path}")

        if not isinstance(data['nc'], int) or data['nc'] <= 0:
            raise ValueError(f"El número de clases (nc) debe ser un entero positivo: {data['nc']}")

        if not isinstance(data['names'], list):
            raise ValueError("Los nombres de clases (names) deben ser una lista")

        if len(data['names']) != data['nc']:
            raise ValueError(
                f"El número de nombres ({len(data['names'])}) no coincide "
                f"con el número de clases ({data['nc']})"
            )

    except yaml.YAMLError as e:
        raise ValueError(f"Error al parsear el archivo YAML {data_yaml_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error al validar el archivo {data_yaml_path}: {e}") from e

    return True


def validate_directory_path(directory_path: str | Path, create: bool = False) -> bool:
    """
    Valida que el directorio existe y es accesible.

    Args:
        directory_path: Ruta al directorio
        create: Si se debe crear el directorio si no existe

    Returns:
        True si el directorio es válido

    Raises:
        FileNotFoundError: Si el directorio no existe y create=False
        PermissionError: Si no hay permisos para acceder al directorio
    """
    directory_path = Path(directory_path)

    if create and not directory_path.exists():
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(f"No se pudo crear el directorio {directory_path}: {e}") from e

    if not directory_path.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory_path}")

    if not directory_path.is_dir():
        raise ValueError(f"No es un directorio válido: {directory_path}")

    if not os.access(directory_path, os.W_OK):
        raise PermissionError(f"No hay permisos de escritura en: {directory_path}")

    return True


def validate_confidence_threshold(confidence: float) -> bool:
    """
    Valida que el umbral de confianza está en el rango correcto.

    Args:
        confidence: Valor del umbral de confianza

    Returns:
        True si el valor es válido

    Raises:
        ValueError: Si el valor está fuera del rango [0, 1]
    """
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"El umbral de confianza debe ser un número: {confidence}")

    if not 0 <= confidence <= 1:
        raise ValueError(f"El umbral de confianza debe estar entre 0 y 1: {confidence}")

    return True


def validate_iou_threshold(iou: float) -> bool:
    """
    Valida que el umbral de IoU está en el rango correcto.

    Args:
        iou: Valor del umbral de IoU

    Returns:
        True si el valor es válido

    Raises:
        ValueError: Si el valor está fuera del rango [0, 1]
    """
    if not isinstance(iou, (int, float)):
        raise ValueError(f"El umbral de IoU debe ser un número: {iou}")

    if not 0 <= iou <= 1:
        raise ValueError(f"El umbral de IoU debe estar entre 0 y 1: {iou}")

    return True
