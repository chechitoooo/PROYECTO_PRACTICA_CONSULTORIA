import pandas as pd
from pathlib import Path
import json
import logging

# 1. Configuración de Logging Profesional (Reemplaza a los prints)
logging.basicConfig(
    level=logging.INFO,                
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Configuración de Rutas utilizando Pathlib de forma robusta
BASE_DIR = Path(__file__).resolve().parent.parent # Asumiendo estructura de proyecto
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" # MEJORA: Separar Raw de Processed
ARTIFACTS_PATH = BASE_DIR / "artifacts"

# Crear directorios si no existen
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_PATH]:
    folder.mkdir(parents=True, exist_ok=True)


def ingest():
    """
    Ingesta de datos crudos, validación estructural y conversión a Parquet.
    """
    logger.info("Iniciando ingesta de datos...")

    features_file = RAW_DATA_DIR / "features.csv"
    targets_file = RAW_DATA_DIR / "targets.csv"

    # Validación de existencia de archivos
    if not features_file.exists() or not targets_file.exists():
        logger.error(f"Archivos no encontrados en {RAW_DATA_DIR}")
        raise FileNotFoundError("No se encontraron los archivos CSV en la ruta especificada.")

    # -------------------------
    # Leer datos
    # -------------------------
    logger.info("Leyendo archivos CSV...")
    X = pd.read_csv(features_file)
    y = pd.read_csv(targets_file)

    logger.info(f"Features cargados: {X.shape[0]} filas, {X.shape[1]} columnas")
    logger.info(f"Targets cargados: {y.shape[0]} filas, {y.shape[1]} columnas")

    # -------------------------
    # Validación estructural
    # -------------------------
    logger.info("Validando integridad estructural...")
    
    if len(X) != len(y):
        error_msg = "Features y Targets tienen diferente número de filas"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if X.empty or y.empty:
        error_msg = "Dataset de Features o Targets está vacío"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # -------------------------
    # Guardar en formato PARQUET
    # -------------------------
    # MEJORA: Guardar en data/processed indica que el dato ha sido "tocado" por el pipeline
    output_features = PROCESSED_DATA_DIR / "features.parquet"
    output_targets = PROCESSED_DATA_DIR / "targets.parquet"
    
    logger.info(f"Guardando datos procesados en {PROCESSED_DATA_DIR}...")
    X.to_parquet(output_features, index=False)
    y.to_parquet(output_targets, index=False)

    # -------------------------
    # Metadata (Clave en MLOps)
    # -------------------------
    logger.info("Generando reporte de metadatos...")
    report = {
        "dataset_info": {
            "n_rows": len(X),
            "n_features": X.shape[1],
            "columns": list(X.columns),
            "target_name": list(y.columns),
        },
        "data_quality": {
            "missing_values_features": X.isnull().sum().to_dict(),
            "missing_values_target": y.isnull().sum().to_dict()
        }
    }

    report_path = ARTIFACTS_PATH / "ingest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Reporte de ingesta guardado en: {report_path}")
    logger.info("Pipeline de ingesta finalizado exitosamente.")


if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        logger.error(f"Pipeline fallido: {e}")
