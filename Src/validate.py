import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import mlflow
import mlflow.sklearn

def train_model():
    # 1. Configurar MLflow para seguimiento de experimentos
    mlflow.set_experiment('adult-income-prediction')
    
    with mlflow.start_run():
        # Cargar datos procesados
        print("Cargando datos para entrenamiento...")
        df = pd.read_parquet('Data/processed/adult_processed.parquet')
        X = df.drop(df.columns[-1], axis=1)
        y = df[df.columns[-1]]
        
        # Dividir el dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Entrenar el modelo definiendo hiperparámetros
        print("Entrenando RandomForest con MLflow...")
        n_estimators = 100
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        # Loggear parámetros en MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("model_type", "RandomForest")
        
        clf.fit(X_train, y_train)
        
        # 3. Loggear métricas y modelo en MLflow
        accuracy = clf.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        # Loggear el modelo como artefacto
        mlflow.sklearn.log_model(clf, "model")
        
        # 4. Guardar localmente también
        os.makedirs('Models', exist_ok=True)
        joblib.dump(clf, 'Models/model.joblib')
        
        print(f"¡Entrenamiento completado! Precisión: {accuracy:.2%}")
        print("Resultados registrados en MLflow.")

if __name__ == "__main__":
    train_model()
