"""
Módulo para manejo de datasets de macroinvertebrados.

Este módulo maneja la descarga, validación y preparación de datasets
desde Roboflow y otras fuentes.
"""

from pathlib import Path
from typing import Any

from roboflow import Roboflow

from config import config
from utils.logger import setup_logger
from utils.validators import validate_directory_path


class DatasetManager:
    """
    Clase para manejar datasets de macroinvertebrados.

    Esta clase encapsula toda la lógica de descarga y gestión
    de datasets desde Roboflow.
    """

    def __init__(self):
        """Inicializa el gestor de datasets."""
        self.logger = setup_logger("dataset_manager")
        self.rf = None
        self.project = None

        # Crear directorios necesarios
        self._setup_directories()

    def _setup_directories(self):
        """Configura los directorios necesarios."""
        directories = ["datasets", "logs"]

        for directory in directories:
            validate_directory_path(directory, create=True)

    def setup_roboflow_connection(self,
                                 api_key: str | None = None,
                                 workspace: str | None = None,
                                 project_name: str | None = None):
        """
        Configura la conexión con Roboflow.

        Args:
            api_key: API key de Roboflow (usa config por defecto)
            workspace: Workspace de Roboflow (usa config por defecto)
            project_name: Nombre del proyecto (usa config por defecto)
        """
        api_key = api_key or config.roboflow_api_key
        workspace = workspace or config.roboflow_workspace
        project_name = project_name or config.roboflow_project

        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY no configurado. Definilo en tu .env o pasalo "
                "explícitamente a setup_roboflow_connection(api_key=...)."
            )

        self.logger.info("🔗 Configurando conexión con Roboflow")
        self.logger.info(f"   - Workspace: {workspace}")
        self.logger.info(f"   - Project: {project_name}")

        try:
            # Inicializar Roboflow
            self.rf = Roboflow(api_key=api_key)

            # Obtener workspace y proyecto
            workspace_obj = self.rf.workspace(workspace)
            self.project = workspace_obj.project(project_name)

            self.logger.info("✅ Conexión con Roboflow establecida")

        except Exception as e:
            self.logger.error(f"❌ Error al conectar con Roboflow: {e}")
            raise

    def get_available_versions(self) -> dict[str, Any]:
        """
        Obtiene las versiones disponibles del dataset.

        Returns:
            Información de las versiones disponibles
        """
        if self.project is None:
            raise ValueError(
                "Conexión con Roboflow no establecida. Use setup_roboflow_connection() primero."
            )

        self.logger.info("🔍 Obteniendo versiones disponibles del dataset...")

        try:
            versions_info = self.project.versions()

            versions_data = []
            for version_info in versions_info:
                version_data = {
                    "version": version_info.version,
                    "name": version_info.name,
                    "created": version_info.created,
                    "images": version_info.images,
                    "splits": version_info.splits
                }
                versions_data.append(version_data)

                self.logger.info(f"   • Versión {version_info.version}: {version_info.name}")
                self.logger.info(f"     - Imágenes: {version_info.images}")
                self.logger.info(f"     - Splits: {version_info.splits}")

            return {
                "total_versions": len(versions_data),
                "versions": versions_data
            }

        except Exception as e:
            self.logger.error(f"❌ Error al obtener versiones: {e}")
            raise

    def download_dataset(self,
                        version: int | None = None,
                        format_type: str = "yolov8",
                        output_dir: str = "datasets") -> dict[str, Any]:
        """
        Descarga el dataset desde Roboflow.

        Args:
            version: Versión específica a descargar (usa la última si no se especifica)
            format_type: Formato del dataset
            output_dir: Directorio de salida

        Returns:
            Información del dataset descargado
        """
        if self.project is None:
            raise ValueError(
                "Conexión con Roboflow no establecida. Use setup_roboflow_connection() primero."
            )

        self.logger.info("📦 Descargando dataset desde Roboflow")
        self.logger.info(f"   - Formato: {format_type}")
        self.logger.info(f"   - Directorio: {output_dir}")

        try:
            # Obtener versiones disponibles
            versions_info = self.project.versions()

            # Determinar versión a descargar
            if version is None:
                version = max([v.version for v in versions_info])
                self.logger.info(f"   - Usando última versión: {version}")
            else:
                self.logger.info(f"   - Usando versión específica: {version}")

            # Verificar que la versión existe
            available_versions = [v.version for v in versions_info]
            if version not in available_versions:
                raise ValueError(
                    f"Versión {version} no disponible. Versiones: {available_versions}"
                )

            # Descargar dataset
            self.logger.info(f"📥 Descargando versión {version}...")
            dataset = self.project.version(version).download(format_type)

            # Mover a directorio de salida si es necesario
            if output_dir != "datasets":
                import shutil
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Mover contenido del dataset
                source_path = Path(dataset.location)
                for item in source_path.iterdir():
                    shutil.move(str(item), str(output_path / item.name))

                dataset.location = str(output_path)

            self.logger.info("✅ Dataset descargado exitosamente")
            self.logger.info(f"   - Ubicación: {dataset.location}")

            # Obtener información del dataset
            dataset_info = self._get_dataset_info(dataset.location)

            return {
                "location": dataset.location,
                "version": version,
                "format": format_type,
                "info": dataset_info
            }

        except Exception as e:
            self.logger.error(f"❌ Error al descargar dataset: {e}")
            raise

    def _get_dataset_info(self, dataset_path: str) -> dict[str, Any]:
        """
        Obtiene información del dataset descargado.

        Args:
            dataset_path: Ruta al dataset

        Returns:
            Información del dataset
        """
        try:
            import yaml

            # Verificar si existe data.yaml
            data_yaml_path = Path(dataset_path) / "data.yaml"

            if data_yaml_path.exists():
                with open(data_yaml_path, encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                return {
                    "classes": data.get("names", []),
                    "class_names": data.get("names", []),
                    "splits": {
                        "train": data.get("train", 0),
                        "valid": data.get("val", 0),
                        "test": data.get("test", 0)
                    },
                    "data_yaml": str(data_yaml_path)
                }
            else:
                # Si no existe data.yaml, generar información básica
                return self._generate_basic_info(dataset_path)

        except Exception as e:
            self.logger.error(f"Error al obtener información del dataset: {e}")
            return {
                "classes": [],
                "class_names": [],
                "splits": {"train": 0, "valid": 0, "test": 0},
                "data_yaml": None
            }

    def _generate_basic_info(self, dataset_path: str) -> dict[str, Any]:
        """
        Genera información básica del dataset.

        Args:
            dataset_path: Ruta al dataset

        Returns:
            Información básica del dataset
        """
        try:
            dataset_path = Path(dataset_path)

            # Contar imágenes en cada split
            train_count = len(list((dataset_path / "train").glob("*.jpg"))) + len(
                list((dataset_path / "train").glob("*.png"))
            )
            val_count = len(list((dataset_path / "valid").glob("*.jpg"))) + len(
                list((dataset_path / "valid").glob("*.png"))
            )
            test_count = len(list((dataset_path / "test").glob("*.jpg"))) + len(
                list((dataset_path / "test").glob("*.png"))
            )

            # Obtener clases desde las etiquetas
            classes = self._get_classes_from_labels(str(dataset_path))

            return {
                "classes": classes,
                "class_names": classes,
                "splits": {
                    "train": train_count,
                    "valid": val_count,
                    "test": test_count
                },
                "data_yaml": None
            }

        except Exception as e:
            self.logger.error(f"Error al generar información básica: {e}")
            return {
                "classes": [],
                "class_names": [],
                "splits": {"train": 0, "valid": 0, "test": 0},
                "data_yaml": None
            }

    def _get_classes_from_labels(self, dataset_path: str) -> list:
        """
        Obtiene las clases desde los archivos de etiquetas.

        Args:
            dataset_path: Ruta al dataset

        Returns:
            Lista de nombres de clases
        """
        try:
            dataset_path = Path(dataset_path)
            classes = set()

            # Buscar en train, valid y test
            for split in ["train", "valid", "test"]:
                labels_path = dataset_path / split / "labels"
                if labels_path.exists():
                    for label_file in labels_path.glob("*.txt"):
                        with open(label_file) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 1:
                                    class_id = int(parts[0])
                                    classes.add(class_id)

            # Convertir a lista ordenada
            class_list = sorted(list(classes))

            # Si no se encontraron clases, usar valores por defecto
            if not class_list:
                self.logger.warning(
                    "No se encontraron clases en las etiquetas, usando valores por defecto"
                )
                return ["Belostomatidae", "Chironomidae", "Coenagrionidae", "Dytiscidae",
                       "Hirudinidae", "Libellulidae", "Noteridae", "Physidae", "Planorbidae"]

            # Mapear IDs a nombres (asumiendo orden)
            default_names = ["Belostomatidae", "Chironomidae", "Coenagrionidae", "Dytiscidae",
                           "Hirudinidae", "Libellulidae", "Noteridae", "Physidae", "Planorbidae"]

            class_names = []
            for class_id in class_list:
                if class_id < len(default_names):
                    class_names.append(default_names[class_id])
                else:
                    class_names.append(f"Class_{class_id}")

            return class_names

        except Exception as e:
            self.logger.error(f"Error al obtener clases desde etiquetas: {e}")
            return ["Belostomatidae", "Chironomidae", "Coenagrionidae", "Dytiscidae",
                   "Hirudinidae", "Libellulidae", "Noteridae", "Physidae", "Planorbidae"]

    def validate_dataset_structure(self, dataset_path: str) -> bool:
        """
        Valida la estructura del dataset descargado.

        Args:
            dataset_path: Ruta al dataset

        Returns:
            True si la estructura es válida
        """
        self.logger.info(f"🔍 Validando estructura del dataset: {dataset_path}")

        try:
            dataset_path = Path(dataset_path)

            # Verificar que existe
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

            # Verificar archivo data.yaml
            data_yaml_path = dataset_path / "data.yaml"
            if not data_yaml_path.exists():
                raise FileNotFoundError(f"Archivo data.yaml no encontrado en: {dataset_path}")

            # Cargar y validar data.yaml
            import yaml
            with open(data_yaml_path, encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Verificar estructura básica
            required_keys = ['train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Clave requerida '{key}' no encontrada en data.yaml")

            # Verificar rutas de datos directamente en la estructura del dataset
            # Ignorar las rutas del data.yaml y buscar en la estructura real
            # Mapeo de claves del YAML a nombres de carpetas reales
            split_mapping = {
                'train': 'train',
                'val': 'valid'  # La clave es 'val' pero la carpeta es 'valid'
            }

            for split_key, folder_name in split_mapping.items():
                # Buscar directamente en la estructura del dataset
                split_images_path = dataset_path / folder_name / "images"
                if not split_images_path.exists():
                    raise FileNotFoundError(
                        f"Ruta de {split_key} no encontrada: {split_images_path}"
                    )

                # Verificar que hay imágenes
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                images = [f for f in split_images_path.iterdir()
                        if f.suffix.lower() in image_extensions]

                if len(images) == 0:
                    raise ValueError(
                        f"No se encontraron imágenes en {split_key}: {split_images_path}"
                    )

                self.logger.info(f"   - {split_key}: {len(images)} imágenes")

            self.logger.info("✅ Estructura del dataset válida")
            self.logger.info(f"   - Clases: {data.get('nc', 'N/A')}")
            self.logger.info(f"   - Nombres: {data.get('names', [])}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error en la estructura del dataset: {e}")
            raise

    def get_dataset_summary(self, dataset_path: str) -> dict[str, Any]:
        """
        Obtiene un resumen del dataset.

        Args:
            dataset_path: Ruta al dataset

        Returns:
            Resumen del dataset
        """
        try:
            # Validar estructura primero
            if not self.validate_dataset_structure(dataset_path):
                raise ValueError("Estructura del dataset inválida")

            # Obtener información básica
            dataset_path = Path(dataset_path)
            train_path = dataset_path / "train"
            val_path = dataset_path / "valid"
            test_path = dataset_path / "test"

            # Contar imágenes
            train_images = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
            val_images = len(list(val_path.glob("*.jpg"))) + len(list(val_path.glob("*.png")))
            test_images = len(list(test_path.glob("*.jpg"))) + len(list(test_path.glob("*.png")))

            # Obtener clases
            classes = self._get_classes_from_labels(dataset_path)

            return {
                "total_images": train_images + val_images + test_images,
                "train_images": train_images,
                "val_images": val_images,
                "test_images": test_images,
                "classes": classes,
                "splits": {
                    "train": train_images,
                    "valid": val_images,
                    "test": test_images
                }
            }

        except Exception as e:
            self.logger.error(f"Error al obtener resumen del dataset: {e}")
            raise

    def generate_data_yaml(self, dataset_path: str) -> str:
        """
        Genera el archivo data.yaml para el dataset.

        Args:
            dataset_path: Ruta al dataset

        Returns:
            Ruta al archivo data.yaml generado
        """
        try:
            # Obtener resumen del dataset
            summary = self.get_dataset_summary(dataset_path)

            # Crear contenido del data.yaml
            data_yaml_content = {
                "path": str(Path(dataset_path).absolute()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(summary["classes"]),
                "names": summary["classes"]
            }

            # Generar ruta del archivo
            data_yaml_path = Path(dataset_path) / "data.yaml"

            # Escribir archivo
            import yaml
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"✅ data.yaml generado: {data_yaml_path}")
            self.logger.info(f"   - Clases: {len(summary['classes'])}")
            self.logger.info(f"   - Imágenes totales: {summary['total_images']}")

            return str(data_yaml_path)

        except Exception as e:
            self.logger.error(f"Error al generar data.yaml: {e}")
            raise
