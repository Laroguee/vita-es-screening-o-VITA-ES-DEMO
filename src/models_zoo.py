"""
Módulo de Entrenamiento y Benchmarking del Model Zoo Clínico
Proyecto VITA-ES | Huawei ICT Competition 2026-2027

Entrena, valida y serializa:
1. Balanced Random Forest (Baseline)
2. Support Vector Machine (Kernel RBF Calibrado)
3. XGBoost Classifier (Gradient Boosting)
4. 1D CNN - VITA-Net (Red Neuronal Convolucional 1D)
5. Modelos auxiliares multimodales: Coimbra (Metabólico) y WDBC (Morfológico)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

# Imports del sistema y entorno
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras

# Imports del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import CLASS_MAPPING, FEATURE_NAMES, load_dataset


def build_1d_cnn_model(input_shape=(9, 1), num_classes=6) -> keras.Model:
    """
    Construye la arquitectura VITA-Net 1D:
    Red convolucional unidimensional ligera diseñada para procesar
    vectores de espectroscopía dieléctrica y evitar sobreajuste en muestras clínicas.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_multimodal_auxiliary_models(models_dir: str = "models") -> Dict:
    """
    Entrena modelos auxiliares para la fusión multimodal con los datasets Coimbra y WDBC:
    1. Coimbra (dataR2.csv): Riesgo metabólico sistémico (IMC, Glucosa, Insulina, etc.)
    2. WDBC (wdbc.data): Riesgo citológico morfológico ante nódulo palpable
    """
    print("\n" + "=" * 70)
    print(" [*] ENTRENANDO MODELOS AUXILIARES DE FUSIÓN MULTIMODAL")
    print("=" * 70)
    os.makedirs(models_dir, exist_ok=True)
    aux_results = {}

    # 1. Dataset Coimbra (Metabólico)
    coimbra_path = os.path.join("data", "dataR2.csv")
    if os.path.exists(coimbra_path):
        df_coimbra = pd.read_csv(coimbra_path)
        coimbra_features = [c for c in df_coimbra.columns if c != "Classification"]
        X_coimbra = df_coimbra[coimbra_features].values
        # 1=Control Sano, 2=Paciente Cáncer -> convertir a 0 y 1
        y_coimbra = (df_coimbra["Classification"].values == 2).astype(int)

        scaler_coimbra = StandardScaler()
        X_coimbra_scaled = scaler_coimbra.fit_transform(X_coimbra)

        model_coimbra = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
        model_coimbra.fit(X_coimbra_scaled, y_coimbra)

        joblib.dump(model_coimbra, os.path.join(models_dir, "model_coimbra.pkl"))
        joblib.dump(scaler_coimbra, os.path.join(models_dir, "scaler_coimbra.pkl"))

        aux_results["coimbra"] = {
            "features": coimbra_features,
            "samples": len(df_coimbra),
            "status": "ready"
        }
        print(f" [+] Modelo Coimbra entrenado y guardado ({len(df_coimbra)} muestras).")

    # 2. Dataset WDBC (Wisconsin Cytology / Morfología)
    wdbc_path = os.path.join("data", "wdbc.data")
    if os.path.exists(wdbc_path):
        df_wdbc = pd.read_csv(wdbc_path, header=None)
        # Col 0: ID, Col 1: Diagnosis (B/M), Col 2-31: 30 features
        y_wdbc = (df_wdbc[1].values == "M").astype(int)
        X_wdbc = df_wdbc.iloc[:, 2:32].values

        scaler_wdbc = StandardScaler()
        X_wdbc_scaled = scaler_wdbc.fit_transform(X_wdbc)

        model_wdbc = LogisticRegression(max_iter=1000, random_state=42)
        model_wdbc.fit(X_wdbc_scaled, y_wdbc)

        joblib.dump(model_wdbc, os.path.join(models_dir, "model_wdbc.pkl"))
        joblib.dump(scaler_wdbc, os.path.join(models_dir, "scaler_wdbc.pkl"))

        aux_results["wdbc"] = {
            "num_features": 30,
            "samples": len(df_wdbc),
            "status": "ready"
        }
        print(f" [+] Modelo WDBC entrenado y guardado ({len(df_wdbc)} muestras).")

    return aux_results


def train_and_benchmark_model_zoo(
    data_path: str = None,
    models_dir: str = "models",
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Ejecuta validación cruzada estratificada (k=5) para comparar:
    - Balanced Random Forest
    - Support Vector Machine (RBF)
    - XGBoost Classifier
    - 1D CNN (VITA-Net 1D)
    
    Genera artefactos serializados y archivo de métricas comparativas models_comparison.json.
    """
    print("=" * 70)
    print(" VITA-ES: BENCHMARKING DE MODEL ZOO Y ENTRENAMIENTO DE IA MÉDICA")
    print(" Huawei ICT Competition 2026-2027 | Topic 1: Innovative AI on Huawei Cloud")
    print("=" * 70)

    # 1. Cargar datos de bioimpedancia
    df = load_dataset(data_path)
    X = df[FEATURE_NAMES].values
    y = df["Class"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)
    car_idx = classes.index("car") if "car" in classes else 0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Definir clasificadores tabulares
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=2.5,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1
        )
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    comparison_results = {}

    # 3. Validar modelos tabulares con Stratified K-Fold
    for name, clf in classifiers.items():
        print(f"\n[*] Evaluando {name} con Stratified K-Fold (k={n_splits})...")
        oof_preds = np.zeros(len(y_encoded), dtype=int)
        oof_probs = np.zeros((len(y_encoded), len(classes)))
        latencies = []

        for train_idx, val_idx in skf.split(X_scaled, y_encoded):
            X_train, y_train = X_scaled[train_idx], y_encoded[train_idx]
            X_val, y_val = X_scaled[val_idx], y_encoded[val_idx]

            clf_fold = clf.__class__(**clf.get_params())
            clf_fold.fit(X_train, y_train)

            t0 = time.perf_counter()
            probs_fold = clf_fold.predict_proba(X_val)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / len(X_val) * 1000.0)

            oof_probs[val_idx] = probs_fold
            oof_preds[val_idx] = np.argmax(probs_fold, axis=1)

        acc = accuracy_score(y_encoded, oof_preds)
        bal_acc = balanced_accuracy_score(y_encoded, oof_preds)
        f1_macro = f1_score(y_encoded, oof_preds, average="macro")
        prec_macro = precision_score(y_encoded, oof_preds, average="macro", zero_division=0)
        rec_macro = recall_score(y_encoded, oof_preds, average="macro", zero_division=0)

        car_true = (y_encoded == car_idx).astype(int)
        car_pred = (oof_preds == car_idx).astype(int)
        car_sens = recall_score(car_true, car_pred, zero_division=0)
        car_prec = precision_score(car_true, car_pred, zero_division=0)
        cm = confusion_matrix(y_encoded, oof_preds).tolist()

        avg_lat = float(np.mean(latencies))

        comparison_results[name] = {
            "accuracy": round(float(acc) * 100, 2),
            "balanced_accuracy": round(float(bal_acc) * 100, 2),
            "f1_macro": round(float(f1_macro), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "carcinoma_sensitivity": round(float(car_sens) * 100, 2),
            "carcinoma_precision": round(float(car_prec) * 100, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "confusion_matrix": cm,
            "architecture_type": "Tabular Ensemble / Kernel"
        }
        print(f"    - Accuracy: {acc*100:.2f}% | Sensibilidad Carcinoma: {car_sens*100:.2f}% | Latencia: {avg_lat:.2f} ms")

    # 4. Validar 1D CNN (VITA-Net 1D)
    print("\n[*] Evaluando 1D CNN (VITA-Net 1D) con Stratified K-Fold (k=5)...")
    X_cnn = X_scaled.reshape((len(X_scaled), len(FEATURE_NAMES), 1))
    oof_preds_cnn = np.zeros(len(y_encoded), dtype=int)
    oof_probs_cnn = np.zeros((len(y_encoded), len(classes)))
    cnn_latencies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cnn, y_encoded)):
        X_train, y_train = X_cnn[train_idx], y_encoded[train_idx]
        X_val, y_val = X_cnn[val_idx], y_encoded[val_idx]

        model_cnn = build_1d_cnn_model(input_shape=(len(FEATURE_NAMES), 1), num_classes=len(classes))
        
        # Entrenamiento rápido y convergente para N=106
        model_cnn.fit(
            X_train, y_train,
            epochs=35,
            batch_size=16,
            verbose=0,
            validation_data=(X_val, y_val)
        )

        t0 = time.perf_counter()
        probs_val = model_cnn.predict(X_val, verbose=0)
        t1 = time.perf_counter()
        cnn_latencies.append((t1 - t0) / len(X_val) * 1000.0)

        oof_probs_cnn[val_idx] = probs_val
        oof_preds_cnn[val_idx] = np.argmax(probs_val, axis=1)

    acc_cnn = accuracy_score(y_encoded, oof_preds_cnn)
    bal_acc_cnn = balanced_accuracy_score(y_encoded, oof_preds_cnn)
    f1_macro_cnn = f1_score(y_encoded, oof_preds_cnn, average="macro")
    prec_macro_cnn = precision_score(y_encoded, oof_preds_cnn, average="macro", zero_division=0)
    rec_macro_cnn = recall_score(y_encoded, oof_preds_cnn, average="macro", zero_division=0)

    car_true = (y_encoded == car_idx).astype(int)
    car_pred_cnn = (oof_preds_cnn == car_idx).astype(int)
    car_sens_cnn = recall_score(car_true, car_pred_cnn, zero_division=0)
    car_prec_cnn = precision_score(car_true, car_pred_cnn, zero_division=0)
    cm_cnn = confusion_matrix(y_encoded, oof_preds_cnn).tolist()
    avg_lat_cnn = float(np.mean(cnn_latencies))

    comparison_results["1D CNN (VITA-Net)"] = {
        "accuracy": round(float(acc_cnn) * 100, 2),
        "balanced_accuracy": round(float(bal_acc_cnn) * 100, 2),
        "f1_macro": round(float(f1_macro_cnn), 4),
        "precision_macro": round(float(prec_macro_cnn), 4),
        "recall_macro": round(float(rec_macro_cnn), 4),
        "carcinoma_sensitivity": round(float(car_sens_cnn) * 100, 2),
        "carcinoma_precision": round(float(car_prec_cnn) * 100, 2),
        "avg_latency_ms": round(avg_lat_cnn, 2),
        "confusion_matrix": cm_cnn,
        "architecture_type": "Deep Learning Conv1D"
    }
    print(f"    - Accuracy: {acc_cnn*100:.2f}% | Sensibilidad Carcinoma: {car_sens_cnn*100:.2f}% | Latencia: {avg_lat_cnn:.2f} ms")

    # 5. Entrenar y guardar todos los modelos finales sobre el 100% de los datos
    print("\n[*] Serializando modelos finales para despliegue Edge y ModelArts...")
    os.makedirs(models_dir, exist_ok=True)

    # Random Forest
    rf_final = classifiers["Random Forest"]
    rf_final.fit(X_scaled, y_encoded)
    joblib.dump(rf_final, os.path.join(models_dir, "model_rf.pkl"))
    joblib.dump(rf_final, os.path.join(models_dir, "model.pkl"))  # Compatibilidad

    # SVM
    svm_final = classifiers["SVM (RBF)"]
    svm_final.fit(X_scaled, y_encoded)
    joblib.dump(svm_final, os.path.join(models_dir, "model_svm.pkl"))

    # XGBoost
    xgb_final = classifiers["XGBoost"]
    xgb_final.fit(X_scaled, y_encoded)
    joblib.dump(xgb_final, os.path.join(models_dir, "model_xgb.pkl"))

    # 1D CNN
    final_cnn = build_1d_cnn_model(input_shape=(len(FEATURE_NAMES), 1), num_classes=len(classes))
    final_cnn.fit(X_cnn, y_encoded, epochs=45, batch_size=16, verbose=0)
    final_cnn.save(os.path.join(models_dir, "model_cnn1d.keras"))

    # Scaler y LabelEncoder
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))

    # 6. Modelos Auxiliares Multimodales
    aux_results = train_multimodal_auxiliary_models(models_dir)

    # 7. Metadatos consolidados
    comparison_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(df),
        "classes": classes,
        "class_mapping": CLASS_MAPPING,
        "features": FEATURE_NAMES,
        "models": comparison_results,
        "auxiliary_models": aux_results,
        "best_carcinoma_sensitivity_model": max(
            comparison_results.keys(),
            key=lambda k: comparison_results[k]["carcinoma_sensitivity"]
        ),
        "huawei_hardware_target": {
            "edge_accelerator": "Huawei Ascend 310 / Atlas 200 (MindSpore Lite)",
            "cloud_trainer": "Huawei Cloud ModelArts (Ascend 910 Pro NPU)",
            "recommended_inference_mode": "Edge-First con fallback a ModelArts Cloud"
        }
    }

    with open(os.path.join(models_dir, "models_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Proceso completado exitosamente.")
    print(f"    - Mejor modelo en Sensibilidad Carcinoma: {comparison_metadata['best_carcinoma_sensitivity_model']}")
    print(f"    - Artefactos guardados en '{models_dir}/'.")
    print("=" * 70)

    return comparison_metadata


if __name__ == "__main__":
    train_and_benchmark_model_zoo()
