import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

def evaluate():
    # 1. Cargar el modelo binario (el que ves con símbolos raros)
    model = joblib.load('models/model.joblib')
    
    # 2. Cargar datos de prueba (ya procesados)
    df = pd.read_parquet('Data/processed/adult_processed.parquet')
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    
    # 3. Hacer predicciones
    predictions = model.predict(X)
    
    # 4. Mostrar resultados humanos
    print("--- Reporte del Modelo ---")
    print(f"Precisión General: {accuracy_score(y, predictions):.2%}")
    print("\nDetalle por clase:")
    print(classification_report(y, predictions))

if __name__ == "__main__":
    evaluate()
