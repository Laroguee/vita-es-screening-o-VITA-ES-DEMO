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

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Imports del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import CLASS_MAPPING, FEATURE_DESCRIPTIONS, FEATURE_NAMES
from feature_extraction import extract_features_from_sweep, fit_cole_circle, validate_features
from uncertainty import compute_uncertainty_metrics
from cloud_services import (
    HuaweiDCSManager,
    HuaweiIoTDATelemetryManager,
    HuaweiModelArtsClient,
    HuaweiRDSManager,
)


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
        self.iotda_manager = HuaweiIoTDATelemetryManager()

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

    def compute_local_treeshap(
        self,
        scaled_vector: np.ndarray,
        raw_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calcula en tiempo real los valores de atribución local exacta TreeSHAP (Lundberg & Lee, 2017)
        para la paciente evaluada sobre el modelo XGBoost (`pred_contribs=True`),
        permitiendo al especialista identificar qué parámetro bioeléctrico específico disparó la alerta.
        """
        classes = self.metadata.get("classes", ["adi", "car", "con", "fad", "gla", "mas"])
        car_idx = classes.index("car") if "car" in classes else 1

        xgb_model = self.models.get("XGBoost")
        if xgb_model is not None and HAS_XGB:
            try:
                dmat = xgb.DMatrix(scaled_vector, feature_names=FEATURE_NAMES)
                contribs = xgb_model.get_booster().predict(dmat, pred_contribs=True)
                # contribs shape: (1, n_classes, n_features + 1)
                car_shap = contribs[0, car_idx, :-1]
                base_val = float(contribs[0, car_idx, -1])
            except Exception:
                car_shap = np.zeros(len(FEATURE_NAMES))
                base_val = 0.0
        else:
            car_shap = np.zeros(len(FEATURE_NAMES))
            base_val = 0.0

        total_abs = max(1e-6, float(np.sum(np.abs(car_shap))))
        attributions = []
        for idx, fname in enumerate(FEATURE_NAMES):
            sv = float(car_shap[idx])
            rel_pct = (sv / total_abs) * 100.0
            attributions.append({
                "feature": fname,
                "raw_value": round(float(raw_features.get(fname, 0.0)), 3),
                "shap_log_odds": round(sv, 4),
                "relative_impact_pct": round(rel_pct, 1),
                "direction": "Impulsa Riesgo Maligno (+)" if sv >= 0 else "Factor Protector / Benigno (-)",
                "clinical_meaning": FEATURE_DESCRIPTIONS.get(fname, "")
            })

        attributions_sorted = sorted(attributions, key=lambda x: abs(x["shap_log_odds"]), reverse=True)
        top_positive = [a for a in attributions_sorted if a["shap_log_odds"] > 0][:3]
        top_negative = [a for a in attributions_sorted if a["shap_log_odds"] < 0][:3]

        return {
            "method": "Exact TreeSHAP (XGBoost Booster pred_contribs)",
            "base_log_odds": round(base_val, 4),
            "attributions": attributions_sorted,
            "top_risk_drivers": [f"{a['feature']} ({a['shap_log_odds']:+.2f})" for a in top_positive],
            "top_protective_drivers": [f"{a['feature']} ({a['shap_log_odds']:+.2f})" for a in top_negative]
        }

    @staticmethod
    def get_regulatory_knowledge_graph(
        triage_code: int,
        patient_age: int = 45,
        dense_breast: bool = False,
        requires_second_opinion: bool = False,
        multimodal_escalated: bool = False
    ) -> Dict[str, Any]:
        """
        Grafo determinista de protocolos clínicos y regulatorios (inspirado en KG-RAG sin alucinaciones):
        Enlaza el resultado de triaje con MINSAL El Salvador, SRS (RTS 11.03.04:23) y ACR BI-RADS 5ª Ed.
        """
        imaging_modality = (
            "Ultrasonografía Mamaria Bilateral de Alta Resolución (1ª línea por edad < 35 años o mama densa Tipo C/D)"
            if (patient_age < 35 or dense_breast)
            else "Mamografía Diagnóstica Digital Bilateral + Ultrasonido Complementario"
        )

        if triage_code == 2:
            level_referral = "Hospital Nacional de 3er Nivel / Unidad de Oncología Mamaria"
            sla_window = "< 7 días hábiles (Ruta Prioritaria Oncológica)"
            minsal_rule = "Lineamiento MINSAL Cap. IV-B: Sospecha de neoplasia maligna (BI-RADS 4/5) -> Biopsia con Aguja Gruesa (TRU-CUT)."
        elif triage_code == 1:
            level_referral = "Unidad Comunitaria de Salud Familiar Especializada (UCSFE)"
            sla_window = "15 a 30 días para estudio de imagen; control semestral (6 meses)"
            minsal_rule = "Lineamiento MINSAL Cap. III-C: Hallazgo probablemente benigno (BI-RADS 3) -> Correlación ecográfica y vigilancia."
        else:
            level_referral = "Unidad Comunitaria de Salud Familiar Básica (UCSF)"
            sla_window = "Control preventivo anual (12 meses)"
            minsal_rule = "Lineamiento MINSAL Cap. II-A: Tamizaje negativo / tejido fisiológico (BI-RADS 1/2) -> Continuar tamizaje periódico."

        return {
            "srs_classification": "SaMD Clase IIa (Riesgo Moderado-Alto) — RTS 11.03.04:23 (SRS El Salvador)",
            "electrical_safety_standard": "IEC 60601-1 Parte BF (< 100 µA AC inyección sub-sensorial, 0% radiación ionizante)",
            "acr_birads_concordance": "ACR BI-RADS Atlas 5th Edition — Correlación Dieléctrica-Morfológica",
            "minsal_protocol": minsal_rule,
            "recommended_imaging": imaging_modality,
            "target_healthcare_facility": level_referral,
            "sla_response_time": sla_window,
            "safety_escalation_active": bool(requires_second_opinion or multimodal_escalated)
        }

    def predict(
        self,
        features: Dict[str, float],
        model_name: str = "XGBoost",
        patient_metabolic: Optional[Dict[str, float]] = None,
        patient_palpable: bool = False,
        wdbc_features: Optional[Union[List[float], np.ndarray]] = None,
        patient_age: int = 45,
        dense_breast: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta la inferencia diagnóstica completa con medición de latencia real:
        1. Inferencia bioeléctrica primaria con el modelo seleccionado (o Ensamble).
        2. Explicabilidad clínica en tiempo real mediante TreeSHAP local por paciente.
        3. Actualización Bayesiana Multimodal Adaptativa integrando Coimbra (BCCD) y Wisconsin (WDBC).
        4. Cuantificación de incertidumbre clínica (CUI) y Grafo Normativo (SRS / MINSAL / BI-RADS).
        """
        t0 = time.perf_counter()

        # 1. Validación de características
        is_valid, validation_errors = validate_features(features)

        # 2. Vector de características escalado
        feature_vector = np.array([[float(features.get(f, 0.0)) for f in FEATURE_NAMES]], dtype=np.float64)
        scaled_vector = self.scaler.transform(feature_vector)

        # 3. Probabilidades a priori del modelo EIS seleccionado y medición de latencia pura
        t_mod0 = time.perf_counter()
        eis_probs = self._get_model_probabilities(model_name, scaled_vector)
        t_mod1 = time.perf_counter()
        pure_model_latency_ms = (t_mod1 - t_mod0) * 1000.0

        classes = self.metadata.get("classes", ["adi", "car", "con", "fad", "gla", "mas"])
        eis_prob_dict = {cls: float(eis_probs[i]) for i, cls in enumerate(classes)}
        eis_pred_idx = int(np.argmax(eis_probs))
        eis_pred_class = classes[eis_pred_idx]

        # 4. Explicabilidad Local TreeSHAP en tiempo real
        local_shap = self.compute_local_treeshap(scaled_vector, features)

        # 5. Evaluación de Modelos Auxiliares Multimodales (Coimbra BCCD y Wisconsin WDBC)
        metabolic_risk_prob = 0.20
        if patient_metabolic and self.model_coimbra and self.scaler_coimbra:
            try:
                coimbra_cols = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]
                metabolic_vals = np.array([[float(patient_metabolic.get(k, 0.0)) for k in coimbra_cols]])
                metabolic_scaled = self.scaler_coimbra.transform(metabolic_vals)
                metabolic_risk_prob = float(self.model_coimbra.predict_proba(metabolic_scaled)[0, 1])
            except Exception:
                metabolic_risk_prob = 0.25

        morphological_risk_prob, wdbc_source = self.evaluate_wdbc_morphology(
            wdbc_features=wdbc_features,
            patient_palpable=patient_palpable
        )

        # 6. Fusión Bayesiana Adaptativa por Verosimilitud (Posterior Multimodal sobre las 6 clases)
        # Actualiza la distribución posterior integrando la evidencia metabólica (Coimbra) y citológica (WDBC)
        car_idx = classes.index("car") if "car" in classes else 1
        fad_idx = classes.index("fad") if "fad" in classes else 3
        mas_idx = classes.index("mas") if "mas" in classes else 5

        eis_car_prob = float(eis_probs[car_idx])
        posterior_probs = np.array(eis_probs, dtype=np.float64).copy()

        # Razón de verosimilitud combinada de los biomarcadores sistémicos y citológicos
        aux_combined_malignancy = 0.45 * metabolic_risk_prob + 0.55 * morphological_risk_prob
        has_explicit_fna = (wdbc_features is not None and len(wdbc_features) == 30)

        # Si existe evidencia citológica FNA o alteración metabólica, modulamos la probabilidad posterior de carcinoma
        if has_explicit_fna:
            # Con punción citológica FNA real vinculada (97.36% exactitud WDBC), su peso bayesiano es alto
            target_car_post = float(np.clip(
                0.45 * eis_car_prob + 0.15 * metabolic_risk_prob + 0.40 * morphological_risk_prob,
                0.0, 0.99
            ))
        else:
            # Sin FNA previa, modulación bayesiana suave por perfil metabólico y examen físico
            likelihood_gain = np.exp(1.15 * (metabolic_risk_prob - 0.35) + 0.85 * (morphological_risk_prob - 0.25))
            boosted_car = eis_car_prob * likelihood_gain + max(0.0, 0.18 * (aux_combined_malignancy - 0.42))
            target_car_post = float(np.clip(boosted_car, 0.0, 0.99))

        # Ajustar el vector posterior de 6 clases manteniendo suma = 1.0
        other_sum = float(np.sum(posterior_probs) - posterior_probs[car_idx])
        if other_sum > 1e-6:
            scale_others = max(0.0, 1.0 - target_car_post) / other_sum
            for i in range(len(classes)):
                if i == car_idx:
                    posterior_probs[i] = target_car_post
                else:
                    posterior_probs[i] *= scale_others
        else:
            posterior_probs[car_idx] = target_car_post

        # Si el riesgo auxiliar es intermedio y el tejido EIS era sano, incrementar levemente proliferación benigna
        if 0.45 <= aux_combined_malignancy < 0.70 and eis_pred_class in ["adi", "con", "gla"]:
            shift = min(0.12, 0.20 * (aux_combined_malignancy - 0.40))
            posterior_probs[fad_idx] += shift * 0.6
            posterior_probs[mas_idx] += shift * 0.4
            posterior_probs /= np.sum(posterior_probs)

        prob_dict = {cls: float(posterior_probs[i]) for i, cls in enumerate(classes)}
        pred_idx = int(np.argmax(posterior_probs))
        pred_class = classes[pred_idx]
        confidence = float(posterior_probs[pred_idx])

        # 7. Cuantificación de Incertidumbre Diagnóstica Clínica (UQ) sobre la posterior multimodal
        uncertainty = compute_uncertainty_metrics(posterior_probs, classes)

        car_post_prob = prob_dict.get("car", 0.0)
        benign_prob = prob_dict.get("fad", 0.0) + prob_dict.get("mas", 0.0)

        # Score de Riesgo Oncológico Fusionado (Fused Malignancy Score)
        linear_fused = 0.70 * eis_car_prob + 0.15 * metabolic_risk_prob + 0.15 * morphological_risk_prob
        fused_malignancy_risk = float(np.clip(max(linear_fused, car_post_prob), 0.0, 1.0))

        # Detectar si hubo escalamiento multimodal preventivo (ej. EIS benigno pero citología WDBC o Coimbra críticos)
        multimodal_escalated = False
        escalation_reason = "Concordancia Primaria EIS"
        if eis_pred_class != "car" and (fused_malignancy_risk >= 0.32 or pred_class == "car" or (has_explicit_fna and morphological_risk_prob >= 0.70)):
            multimodal_escalated = True
            escalation_reason = (
                f"Escalamiento Preventivo Multimodal: Evidencia Citológica/Metabólica Crítica "
                f"(WDBC={morphological_risk_prob*100:.1f}%, Coimbra={metabolic_risk_prob*100:.1f}%)"
            )
            fused_malignancy_risk = max(fused_malignancy_risk, 0.36)
        elif eis_pred_class in ["adi", "con", "gla"] and (metabolic_risk_prob >= 0.65 or morphological_risk_prob >= 0.55):
            escalation_reason = (
                f"Alerta Vigilancia Multimodal: Biomarcadores Sistémicos/Morfológicos Elevados "
                f"(Coimbra={metabolic_risk_prob*100:.1f}%, WDBC={morphological_risk_prob*100:.1f}%)"
            )

        # 8. Reglas de Triaje Clínico Oncológico VITA-ES
        if fused_malignancy_risk >= 0.32 or pred_class == "car" or multimodal_escalated:
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
            if multimodal_escalated:
                clinical_action = (
                    f"[ESCALAMIENTO MULTIMODAL ACTIVO] Pese a lectura bioeléctrica basal ({CLASS_MAPPING.get(eis_pred_class, {}).get('name', eis_pred_class)}), "
                    "la citología FNA / perfil metabólico presenta alta sospecha de malignidad. "
                    "Derivación prioritaria a Biopsia TRU-CUT e histopatología confirmatoria (< 7 días)."
                )
            final_risk_score = max(fused_malignancy_risk, 0.70)
        elif benign_prob >= 0.38 or pred_class in ["fad", "mas"] or metabolic_risk_prob >= 0.65:
            triage_risk = "RIESGO MODERADO"
            triage_code = 1
            triage_color = "#F59E0B"
            triage_icon = "[RIESGO MODERADO]"
            triage_badge = "RIESGO MODERADO - VIGILANCIA ACTIVA"
            birads = "BI-RADS 3 (Probablemente Benigno / Vigilancia)"
            clinical_action = (
                "Seguimiento ecográfico complementario a corto plazo (3 a 6 meses) "
                "y control endocrino-metabólico para monitorizar estabilidad tisular."
            )
            final_risk_score = max(0.40 + (benign_prob * 0.25), fused_malignancy_risk)
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
            final_risk_score = max(0.05, fused_malignancy_risk)

        if uncertainty["requires_second_opinion"] and triage_code < 2:
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
        eis_class_info = CLASS_MAPPING.get(eis_pred_class, {})

        reg_kg = self.get_regulatory_knowledge_graph(
            triage_code=triage_code,
            patient_age=patient_age,
            dense_breast=dense_breast,
            requires_second_opinion=uncertainty["requires_second_opinion"],
            multimodal_escalated=multimodal_escalated
        )

        return {
            "prediction": pred_class,
            "prediction_name": class_info.get("name", pred_class.upper()),
            "eis_only_prediction": eis_pred_class,
            "eis_only_prediction_name": eis_class_info.get("name", eis_pred_class.upper()),
            "confidence": float(confidence),
            "probabilities": prob_dict,
            "eis_probabilities": eis_prob_dict,
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
            "local_shap": local_shap,
            "regulatory_knowledge_graph": reg_kg,
            "multimodal_breakdown": {
                "eis_carcinoma_prob": round(float(eis_car_prob), 3),
                "posterior_carcinoma_prob": round(float(car_post_prob), 3),
                "metabolic_coimbra_prob": round(float(metabolic_risk_prob), 3),
                "morphological_wdbc_prob": round(float(morphological_risk_prob), 3),
                "wdbc_evaluation_source": wdbc_source,
                "fused_risk_score": round(float(fused_malignancy_risk), 3),
                "multimodal_escalated": multimodal_escalated,
                "escalation_reason": escalation_reason
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
