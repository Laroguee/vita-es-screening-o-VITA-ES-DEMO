"""
Módulo de Entrenamiento y Validación de Modelos de IA Médica para Bioimpedancia
Proyecto VITA-ES | Huawei ICT Competition 2026-2027
Topic 1: Creating Innovative AI Applications Powered by Huawei Cloud
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importar funciones del proyecto
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import CLASS_MAPPING, FEATURE_NAMES, load_dataset


def train_model(
    data_path: str = None,
    models_dir: str = "models",
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Entrena y valida el clasificador de bioimpedancia mamaria con validación cruzada estratificada.
    Genera artefactos listos para despliegue en Huawei Cloud ModelArts y edge inference.
    """
    print("=" * 70)
    print(" VITA-ES: PIPELINE DE ENTRENAMIENTO DE IA MÉDICA (HUAWEI MODELARTS)")
    print("=" * 70)

    # 1. Cargar y preparar datos
    df = load_dataset(data_path)
    print(f"[*] Registros cargados: {len(df)}")
    print(f"[*] Distribución de clases:\n{df['Class'].value_counts().to_string()}")

    X = df[FEATURE_NAMES].values
    y = df["Class"].values

    # 2. Codificación y escalamiento
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Validación Cruzada Estratificada (Stratified K-Fold)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Clasificador: Random Forest calibrado con ponderación balanceada para mitigar
    # el impacto crítico de falsos negativos en tejido maligno (carcinoma)
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1
    )

    print("\n[*] Ejecutando validación cruzada estratificada (k=5)...")
    cv_preds = cross_val_predict(classifier, X_scaled, y_encoded, cv=skf)
    cv_probs = cross_val_predict(classifier, X_scaled, y_encoded, cv=skf, method="predict_proba")

    # 4. Métricas clínicas globales y específicas
    acc = accuracy_score(y_encoded, cv_preds)
    bal_acc = balanced_accuracy_score(y_encoded, cv_preds)
    f1_macro = f1_score(y_encoded, cv_preds, average="macro")
    precision_macro = precision_score(y_encoded, cv_preds, average="macro")
    recall_macro = recall_score(y_encoded, cv_preds, average="macro")

    # Métrica de Carcinoma (Sensibilidad Oncológica Crítica)
    car_idx = classes.index("car") if "car" in classes else 0
    car_true_binary = (y_encoded == car_idx).astype(int)
    car_pred_binary = (cv_preds == car_idx).astype(int)
    car_sensitivity = recall_score(car_true_binary, car_pred_binary)
    car_precision = precision_score(car_true_binary, car_pred_binary)

    # Matriz de confusión
    cm = confusion_matrix(y_encoded, cv_preds).tolist()

    print(f"\n[+] Métricas de Validación Cruzada:")
    print(f"    - Exactitud Global (Accuracy) : {acc * 100:.2f}%")
    print(f"    - Exactitud Balanceada        : {bal_acc * 100:.2f}%")
    print(f"    - F1-Score Macro              : {f1_macro:.4f}")
    print(f"    - Sensibilidad Carcinoma (Car): {car_sensitivity * 100:.2f}% (CRÍTICO CLÍNICO)")
    print(f"    - Precisión Carcinoma (Car)   : {car_precision * 100:.2f}%")

    # 5. Ajustar modelo final sobre todos los datos
    classifier.fit(X_scaled, y_encoded)

    # Importancia de características
    feature_importances = dict(zip(FEATURE_NAMES, classifier.feature_importances_.tolist()))
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    print("\n[+] Jerarquía de importancia de características biofísicas:")
    for feat, imp in sorted_importances:
        print(f"    - {feat:8s}: {imp * 100:5.2f}%")

    # 6. Guardar artefactos en directorio models/
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "metadata.json")

    joblib.dump(classifier, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_name": "VITA-ES-EIS-Classifier",
        "version": "1.0.0",
        "framework": "scikit-learn / Huawei ModelArts compatible",
        "algorithm": "Balanced Random Forest Ensemble",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(df),
        "features": FEATURE_NAMES,
        "classes": classes,
        "class_mapping": CLASS_MAPPING,
        "metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "f1_macro": float(f1_macro),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "carcinoma_sensitivity": float(car_sensitivity),
            "carcinoma_precision": float(car_precision)
        },
        "confusion_matrix": cm,
        "feature_importances": feature_importances,
        "huawei_cloud_deployment": {
            "modelarts_service_id": "serv-vita-es-edge-infer-v1",
            "model_format": "pkl",
            "runtime_env": "Python 3.12 (Ascend/x86 compatible)",
            "api_endpoint": "https://apigw.cn-north-4.huaweicloudapis.com/v1/infers/vita-es-triage",
            "recommended_instance": "modelarts.vm.cpu.2u4g / ECS c7.large.2"
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Artefactos guardados exitosamente en '{models_dir}/':")
    print(f"    - {model_path}")
    print(f"    - {scaler_path}")
    print(f"    - {meta_path}")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    train_model()
