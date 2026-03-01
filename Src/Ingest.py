# src/ingest.py

import pandas as pd
import json
import os
!pip install ucimlrepo

from pathlib import Path

def ingest_adult(output_dir: str = 'data/raw') -> dict:
    # 1. Fetch (Descarga)
    adult = fetch_ucirepo(id=2)
    
    # 2. Preparar ruta de almacenamiento
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # 3. Almacenar (Formato Parquet para eficiencia)
    adult.data.features.to_parquet(path / 'features.parquet')
    adult.data.targets.to_parquet(path / 'targets.parquet')
    
    # 4. Retornar metadatos para validación
    return {
        'n_rows': len(adult.data.features),
        'n_features': adult.data.features.shape[1],
        'target_dist': adult.data.targets.value_counts().to_dict()
    }

if __name__ == "__main__":
    stats = ingest_adult()
    print(f"Ingesta completada: {stats}")
