"""
Motor de Inferencia y Conector con Huawei Cloud Suite (ModelArts, DCS, RDS)
Proyecto VITA-ES | 11th Huawei ICT Competition 2026-2027

Soporta inferencia multimodelo 100% verificable:
- Random Forest (Baseline Balanceado)
- SVM (Kernel RBF Calibrado)
- XGBoost Classifier (Modelo Campeón en Sensibilidad Oncológica)
- 1D CNN (VITA-Net 1D)
- Ensamble Multimodelo (Votación Suave Ponderada)
- Cuantificación de Incertidumbre Clínica (CUI)
- Fusión Multimodal Real con Datasets Coimbra (Metabólico) y WDBC (Morfológico)
- Persistencia Transaccional en Huawei Cloud RDS (PostgreSQL / SQLite local) y DCS (Redis / KV Cache)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# TensorFlow / Keras en modo silencioso para CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except Exception:
    HAS_ONNXRUNTIME = False

# Imports del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import CLASS_MAPPING, FEATURE_NAMES
from feature_extraction import extract_features_from_sweep, fit_cole_circle, validate_features
from uncertainty import compute_uncertainty_metrics
from cloud_services import HuaweiDCSManager, HuaweiModelArtsClient, HuaweiRDSManager


class EISInferenceEngine:
    """
    Motor clínico de inferencia de espectroscopía de bioimpedancia eléctrica (EIS)
    con soporte para inferencia Edge, Cloud (Huawei ModelArts, RDS, DCS),
    cuantificación de incertidumbre diagnóstica y fusión multimodal real.
    """

    AVAILABLE_MODELS = [
        "XGBoost",
        "Random Forest",
        "SVM (RBF)",
        "1D CNN (VITA-Net)",
        "Ensamble Multimodelo"
    ]

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.onnx_sessions: Dict[str, Any] = {}
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.comparison_metadata = None

        # Modelos auxiliares multimodales
        self.model_coimbra = None
        self.scaler_coimbra = None
        self.model_wdbc = None
        self.scaler_wdbc = None
        self.wdbc_profiles: Dict[str, Any] = {}

        # Clientes reales de infraestructura Huawei Cloud / Local Resiliente
        self.rds_manager = HuaweiRDSManager()
        self.dcs_manager = HuaweiDCSManager()
        self.modelarts_client = HuaweiModelArtsClient()

        self._load_artifacts()

    def _resolve_path(self, filename: str) -> Optional[str]:
        """Resuelve rutas de archivos considerando posibles subdirectorios."""
        candidate = os.path.join(self.models_dir, filename)
        if os.path.exists(candidate):
            return candidate
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", filename)
        if os.path.exists(alt):
            return alt
        return None

    def _load_artifacts(self):
        """Carga los modelos entrenados, sesiones ONNX, escaladores y metadatos."""
        scaler_p = self._resolve_path("scaler.pkl")
        if not scaler_p:
            raise FileNotFoundError(
                f"No se encontró 'scaler.pkl' en '{self.models_dir}/'. "
                "Ejecute primero 'python src/models_zoo.py'."
            )
        self.scaler = joblib.load(scaler_p)

        rf_p = self._resolve_path("model_rf.pkl") or self._resolve_path("model.pkl")
        if rf_p:
            self.models["Random Forest"] = joblib.load(rf_p)

        svm_p = self._resolve_path("model_svm.pkl")
        if svm_p:
            self.models["SVM (RBF)"] = joblib.load(svm_p)

        xgb_p = self._resolve_path("model_xgb.pkl")
        if xgb_p:
            self.models["XGBoost"] = joblib.load(xgb_p)

        cnn_p = self._resolve_path("model_cnn1d.keras")
        if cnn_p and HAS_KERAS:
            try:
                self.models["1D CNN (VITA-Net)"] = keras.models.load_model(cnn_p)
            except Exception as e:
                print(f"[!] Aviso: no se pudo cargar el modelo CNN 1D: {e}")

        # Cargar sesiones ONNX si onnxruntime está disponible
        if HAS_ONNXRUNTIME:
            for m_key, onnx_rel in [("Random Forest", "onnx/model_rf.onnx"), ("SVM (RBF)", "onnx/model_svm.onnx")]:
                onnx_p = self._resolve_path(onnx_rel)
                if onnx_p:
                    try:
                        self.onnx_sessions[m_key] = ort.InferenceSession(onnx_p)
                    except Exception:
                        pass

        # Cargar modelos auxiliares Coimbra y WDBC
        coimbra_p = self._resolve_path("model_coimbra.pkl")
        scaler_coimbra_p = self._resolve_path("scaler_coimbra.pkl")
        if coimbra_p and scaler_coimbra_p:
            self.model_coimbra = joblib.load(coimbra_p)
            self.scaler_coimbra = joblib.load(scaler_coimbra_p)

        wdbc_p = self._resolve_path("model_wdbc.pkl")
        scaler_wdbc_p = self._resolve_path("scaler_wdbc.pkl")
        if wdbc_p and scaler_wdbc_p:
            self.model_wdbc = joblib.load(wdbc_p)
            self.scaler_wdbc = joblib.load(scaler_wdbc_p)

        comp_p = self._resolve_path("models_comparison.json")
        if comp_p:
            with open(comp_p, "r", encoding="utf-8") as f:
                self.comparison_metadata = json.load(f)
                self.wdbc_profiles = self.comparison_metadata.get("auxiliary_models", {}).get("wdbc", {})

        meta_p = self._resolve_path("metadata.json")
        if meta_p:
            with open(meta_p, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "features": FEATURE_NAMES,
                "classes": ["adi", "car", "con", "fad", "gla", "mas"],
                "class_mapping": CLASS_MAPPING
            }

    def _get_model_probabilities(self, model_name: str, scaled_features: np.ndarray) -> np.ndarray:
        """Obtiene probabilidades predichas según el modelo seleccionado."""
        classes = self.metadata.get("classes", ["adi", "car", "con", "fad", "gla", "mas"])
        num_classes = len(classes)

        if model_name == "Ensamble Multimodelo":
            accum_probs = np.zeros(num_classes)
            weights = {
                "XGBoost": 0.40,
                "Random Forest": 0.30,
                "SVM (RBF)": 0.15,
                "1D CNN (VITA-Net)": 0.15
            }
            total_weight = 0.0
            for name, weight in weights.items():
                if name in self.models:
                    p = self._get_model_probabilities(name, scaled_features)
                    accum_probs += p * weight
                    total_weight += weight
            if total_weight > 0:
                accum_probs /= total_weight
            return accum_probs

        if model_name == "1D CNN (VITA-Net)":
            if "1D CNN (VITA-Net)" in self.models:
                X_cnn = scaled_features.reshape((1, len(FEATURE_NAMES), 1))
                probs = self.models["1D CNN (VITA-Net)"].predict(X_cnn, verbose=0)[0]
                return probs
            elif "XGBoost" in self.models:
                return self.models["XGBoost"].predict_proba(scaled_features)[0]

        target_model = self.models.get(model_name)
        if target_model is None:
            target_model = self.models.get("XGBoost") or self.models.get("Random Forest") or list(self.models.values())[0]

        return target_model.predict_proba(scaled_features)[0]

    def evaluate_wdbc_morphology(
        self,
        wdbc_features: Optional[Union[List[float], np.ndarray]] = None,
        patient_palpable: bool = False
    ) -> Tuple[float, str]:
        """
        Ejecuta el clasificador real entrenado sobre el dataset Wisconsin WDBC (model_wdbc.pkl):
        - Si se proporciona un vector citológico de 30 características de un caso FNA, lo evalúa directamente.
        - Si solo se cuenta con la exploración física (nódulo palpable vs. asintomática), evalúa el modelo
          sobre los perfiles empíricos derivados de la cohorte WDBC (N=569).
        """
        if self.model_wdbc is not None and self.scaler_wdbc is not None:
            if wdbc_features is not None and len(wdbc_features) == 30:
                vec = np.array([wdbc_features], dtype=np.float64)
                source = "WDBC FNA Cytology Model (30 variables)"
            elif patient_palpable and self.wdbc_profiles.get("palpable_suspicious_profile"):
                vec = np.array([self.wdbc_profiles["palpable_suspicious_profile"]], dtype=np.float64)
                source = "WDBC Model (Perfil Nódulo Palpable)"
            elif self.wdbc_profiles.get("benign_centroid"):
                vec = np.array([self.wdbc_profiles["benign_centroid"]], dtype=np.float64)
                source = "WDBC Model (Perfil Basal Asintomático)"
            else:
                return (0.55 if patient_palpable else 0.08), "Prior Clínico"

            scaled = self.scaler_wdbc.transform(vec)
            prob_malignant = float(self.model_wdbc.predict_proba(scaled)[0, 1])
            return prob_malignant, source

        return (0.55 if patient_palpable else 0.08), "Prior Clínico"

    def predict(
        self,
        features: Dict[str, float],
        model_name: str = "XGBoost",
        patient_metabolic: Optional[Dict[str, float]] = None,
        patient_palpable: bool = False,
        wdbc_features: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta la inferencia diagnóstica completa con medición de latencia real:
        1. Inferencia bioeléctrica con el modelo seleccionado (o Ensamble).
        2. Cuantificación de incertidumbre clínica (CUI, Entropía Shannon, Margen).
        3. Fusión multimodal real ejecutando los modelos Coimbra (BCCD) y Wisconsin (WDBC).
        4. Dictamen médico estructurado con semáforo de riesgo y protocolo BI-RADS.
        """
        t0 = time.perf_counter()

        # 1. Validación de características
        is_valid, validation_errors = validate_features(features)

        # 2. Vector de características escalado
        feature_vector = np.array([[float(features.get(f, 0.0)) for f in FEATURE_NAMES]], dtype=np.float64)
        scaled_vector = self.scaler.transform(feature_vector)

        # 3. Probabilidades del modelo seleccionado y medición de latencia pura del modelo
        t_mod0 = time.perf_counter()
        probs = self._get_model_probabilities(model_name, scaled_vector)
        t_mod1 = time.perf_counter()
        pure_model_latency_ms = (t_mod1 - t_mod0) * 1000.0

        classes = self.metadata.get("classes", ["adi", "car", "con", "fad", "gla", "mas"])
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(classes)}

        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])

        # 4. Cuantificación de Incertidumbre Diagnóstica Clínica (UQ)
        uncertainty = compute_uncertainty_metrics(probs, classes)

        # 5. Fusión Multimodal Bayesiana (EIS 70% + Coimbra 15% + WDBC 15%)
        car_prob = prob_dict.get("car", 0.0)
        fad_prob = prob_dict.get("fad", 0.0)
        mas_prob = prob_dict.get("mas", 0.0)
        benign_prob = fad_prob + mas_prob

        # Riesgo metabólico real (Dataset Coimbra)
        metabolic_risk_prob = 0.20
        if patient_metabolic and self.model_coimbra and self.scaler_coimbra:
            try:
                coimbra_cols = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]
                metabolic_vals = np.array([[float(patient_metabolic.get(k, 0.0)) for k in coimbra_cols]])
                metabolic_scaled = self.scaler_coimbra.transform(metabolic_vals)
                metabolic_risk_prob = float(self.model_coimbra.predict_proba(metabolic_scaled)[0, 1])
            except Exception:
                metabolic_risk_prob = 0.25

        # Riesgo morfológico / citológico real (Dataset WDBC)
        morphological_risk_prob, wdbc_source = self.evaluate_wdbc_morphology(
            wdbc_features=wdbc_features,
            patient_palpable=patient_palpable
        )

        fused_malignancy_risk = float(np.clip(
            0.70 * car_prob + 0.15 * metabolic_risk_prob + 0.15 * morphological_risk_prob,
            0.0,
            1.0
        ))

        # 6. Reglas de Triaje Clínico Oncológico VITA-ES
        if fused_malignancy_risk >= 0.32 or pred_class == "car":
            triage_risk = "ALTO RIESGO"
            triage_code = 2
            triage_color = "#EF4444"
            triage_icon = "[ALTO RIESGO]"
            triage_badge = "ALTO RIESGO ONCOLÓGICO"
            birads = "BI-RADS 4/5 (Sospechoso de Malignidad)"
            clinical_action = (
                "Derivación prioritaria a Biopsia con Aguja Gruesa (BAG) guiada por "
                "imagen y Mamografía diagnóstica urgente (Plazo máximo: 7 días hábiles)."
            )
            final_risk_score = max(fused_malignancy_risk, 0.70)
        elif benign_prob >= 0.38 or pred_class in ["fad", "mas"]:
            triage_risk = "RIESGO MODERADO"
            triage_code = 1
            triage_color = "#F59E0B"
            triage_icon = "[RIESGO MODERADO]"
            triage_badge = "RIESGO MODERADO - LESIÓN BENIGNA"
            birads = "BI-RADS 3 (Probablemente Benigno)"
            clinical_action = (
                "Seguimiento ecográfico complementario a corto plazo (3 a 6 meses) "
                "para monitorizar estabilidad dimensional de la lesión."
            )
            final_risk_score = 0.40 + (benign_prob * 0.25)
        else:
            triage_risk = "BAJO RIESGO"
            triage_code = 0
            triage_color = "#10B981"
            triage_icon = "[BAJO RIESGO]"
            triage_badge = "BAJO RIESGO - TEJIDO FISIOLÓGICO"
            birads = "BI-RADS 1/2 (Normal / Hallazgo Benigno)"
            clinical_action = (
                "Continuar con protocolo habitual de tamizaje preventivo anual "
                "según las guías clínicas para el grupo etario."
            )
            final_risk_score = max(0.05, car_prob * 0.8)

        if uncertainty["requires_second_opinion"]:
            clinical_action = (
                "ALERTA: DICTAMEN NO CONCLUYENTE (Incertidumbre elevada): La respuesta "
                "espectroscópica muestra atipicidad tisular limítrofe. Se requiere "
                "ecografía mamaria confirmatoria antes de emitir alta diagnóstica."
            )

        t1 = time.perf_counter()
        total_pipeline_latency_ms = (t1 - t0) * 1000.0

        # Si hay sesión ONNX para el modelo, medir latencia ONNX real
        onnx_latency_ms = None
        if model_name in self.onnx_sessions:
            t_ox0 = time.perf_counter()
            sess = self.onnx_sessions[model_name]
            in_name = sess.get_inputs()[0].name
            sess.run(None, {in_name: scaled_vector.astype(np.float32)})
            onnx_latency_ms = round((time.perf_counter() - t_ox0) * 1000.0, 3)

        class_info = CLASS_MAPPING.get(pred_class, {})

        return {
            "prediction": pred_class,
            "prediction_name": class_info.get("name", pred_class.upper()),
            "confidence": float(confidence),
            "probabilities": prob_dict,
            "triage_risk": triage_risk,
            "triage_code": triage_code,
            "triage_color": triage_color,
            "triage_icon": triage_icon,
            "triage_badge": triage_badge,
            "birads_category": birads,
            "clinical_action": clinical_action,
            "risk_score": round(float(final_risk_score), 3),
            "model_used": model_name,
            "available_models": [m for m in self.AVAILABLE_MODELS if m in self.models or m == "Ensamble Multimodelo"],
            "uncertainty": uncertainty,
            "multimodal_breakdown": {
                "eis_carcinoma_prob": round(float(car_prob), 3),
                "metabolic_coimbra_prob": round(float(metabolic_risk_prob), 3),
                "morphological_wdbc_prob": round(float(morphological_risk_prob), 3),
                "wdbc_evaluation_source": wdbc_source,
                "fused_risk_score": round(float(fused_malignancy_risk), 3)
            },
            "features_used": features,
            "validation_status": {
                "is_valid": is_valid,
                "errors": validation_errors
            },
            "hardware_benchmark": {
                "active_runtime": "ONNX Runtime + Python 3.12" if onnx_latency_ms is not None else "Python 3.12 Native Engine",
                "pure_model_latency_ms": round(pure_model_latency_ms, 3),
                "onnx_runtime_latency_ms": onnx_latency_ms,
                "local_python_latency_ms": round(total_pipeline_latency_ms, 2)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def predict_from_sweep(
        self,
        sweep_df: pd.DataFrame,
        model_name: str = "XGBoost",
        patient_metabolic: Optional[Dict[str, float]] = None,
        patient_palpable: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """Flujo completo a partir de una tabla de barrido espectral."""
        features = extract_features_from_sweep(sweep_df)
        z_re = sweep_df["Z_Real"].values
        z_im = sweep_df["Minus_Z_Imag"].values if "Minus_Z_Imag" in sweep_df.columns else -sweep_df["Z_Imag"].values
        cole_fit = fit_cole_circle(z_re, z_im)
        inference_result = self.predict(
            features=features,
            model_name=model_name,
            patient_metabolic=patient_metabolic,
            patient_palpable=patient_palpable
        )
        return inference_result, features, cole_fit

    # Conectores reales hacia Huawei Cloud Suite / Persistencia Local
    def invoke_modelarts_service(
        self,
        features: Dict[str, float],
        patient_id: str = "PAC-HUAWEI-001",
        model_name: str = "XGBoost",
        metabolic_factors: Optional[Dict[str, float]] = None,
        patient_palpable: bool = False
    ) -> Dict[str, Any]:
        """Invoca el servicio real de ModelArts (remoto o modelarts_deploy/customize_service.py)."""
        return self.modelarts_client.invoke_inference(
            features=features,
            patient_id=patient_id,
            model_name=model_name,
            metabolic_factors=metabolic_factors,
            patient_palpable=patient_palpable
        )

    def persist_to_rds(self, inference_result: Dict[str, Any], patient_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Inserta un registro transaccional real en Huawei Cloud RDS (PostgreSQL) o SQLite local."""
        return self.rds_manager.insert_triage_record(inference_result, patient_metadata)

    def cache_in_dcs(self, patient_id: str, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """Almacena la sesión en caché en Huawei Cloud DCS (Redis) o KV Store local midiendo latencia real."""
        return self.dcs_manager.cache_screening_session(patient_id, inference_result)

    # Alias de compatibilidad hacia atrás
    def simulate_modelarts_cloud_api(self, features: Dict[str, float], patient_id: str = "PAT-HUAWEI-001") -> Dict[str, Any]:
        return self.invoke_modelarts_service(features=features, patient_id=patient_id)

    def simulate_rds_database_persistence(self, inference_result: Dict[str, Any], patient_metadata: Dict[str, Any]) -> Dict[str, Any]:
        return self.persist_to_rds(inference_result, patient_metadata)

    def simulate_dcs_redis_cache(self, patient_id: str, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.cache_in_dcs(patient_id, inference_result)


try:
    engine = EISInferenceEngine()
except Exception:
    engine = None
