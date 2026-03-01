# src/ingest.py

import pandas as pd
from pathlib import Path
import json
import os
!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

def run_ingestion():
    # 1. Obtener el dataset Adult (ID=2) según tu diapositiva
    print("Descargando dataset desde UCI ML Repository...")
    adult = fetch_ucirepo(id=2)
    
    # 2. Extraer características y objetivos
    X = adult.data.features
    y = adult.data.targets
    
    # 3. Unir en un solo DataFrame para guardarlo
    df_full = pd.concat([X, y], axis=1)
    
    # 4. Crear la ruta de destino (siguiendo tu estructura de proyecto)
    output_path = "data/raw/adult_raw.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 5. Guardar los datos crudos
    df_full.to_csv(output_path, index=False)
    print(f"¡Éxito! Datos guardados en: {output_path}")

if __name__ == "__main__":
    run_ingestion()


