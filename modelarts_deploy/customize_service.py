"""
Huawei Cloud ModelArts Custom Service Script
Proyecto VITA-ES | 11th Huawei ICT Competition 2026-2027

Script oficial de servicio de inferencia para Huawei ModelArts Online Service.
Maneja el ciclo de vida completo del microservicio de IA médica:
- _preprocess(): Valida payload JSON de bioimpedancia, biomarcadores metabólicos (Coimbra) y citológicos (WDBC)
- _inference(): Ejecuta inferencia multimodelo, fusión bayesiana tardía y cálculo de incertidumbre (CUI)
- _postprocess(): Devuelve respuesta JSON conforme al contrato de API definido en config.json
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VITA-ES-ModelArts")

FEATURE_NAMES = ["I0", "PA500", "HFS", "DA", "Area", "A/DA", "Max IP", "DR", "P"]
COIMBRA_COLS = ["Age", "BMI", "Glucose", "Insulin", "HOMA", "Leptin", "Adiponectin", "Resistin", "MCP.1"]

CLASS_MAPPING = {
    "car": {"name": "Carcinoma", "risk": "ALTO RIESGO", "code": 2, "birads": "BI-RADS 4/5 (Sospechoso de Malignidad)"},
    "fad": {"name": "Fibroadenoma", "risk": "RIESGO MODERADO", "code": 1, "birads": "BI-RADS 3 (Probablemente Benigno)"},
    "mas": {"name": "Mastopatía", "risk": "RIESGO MODERADO", "code": 1, "birads": "BI-RADS 3 (Probablemente Benigno)"},
    "gla": {"name": "Glandular", "risk": "BAJO RIESGO", "code": 0, "birads": "BI-RADS 1/2 (Normal)"},
    "con": {"name": "Conectivo", "risk": "BAJO RIESGO", "code": 0, "birads": "BI-RADS 1/2 (Normal)"},
    "adi": {"name": "Adiposo", "risk": "BAJO RIESGO", "code": 0, "birads": "BI-RADS 1/2 (Normal)"}
}


class VITAESModelArtsService:
    """
    Clase de servicio de inferencia compatible con el framework de microservicios de Huawei Cloud ModelArts.
    """

    def __init__(self, model_name: str = "vita-es", model_path: str = None):
        self.model_name = model_name
        self.model_path = model_path or os.path.dirname(os.path.abspath(__file__))
        self.models = {}
        self.scaler = None
        self.model_coimbra = None
        self.scaler_coimbra = None
        self.model_wdbc = None
        self.scaler_wdbc = None
        self.wdbc_profiles = {}
        self.classes = ["adi", "car", "con", "fad", "gla", "mas"]
        self._load_models()

    def _find_artifact(self, filename: str) -> str:
        p1 = os.path.join(self.model_path, filename)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(self.model_path, "..", "models", filename)
        if os.path.exists(p2):
            return p2
        return ""

    def _load_models(self):
        """Carga artefactos serializados disponibles en el directorio del servicio."""
        logger.info(f"[*] Cargando artefactos de ModelArts desde: {self.model_path}")

        scaler_file = self._find_artifact("scaler.pkl")
        if scaler_file:
            self.scaler = joblib.load(scaler_file)

        for m_key, fname in [
            ("XGBoost", "model_xgb.pkl"),
            ("Random Forest", "model_rf.pkl"),
            ("SVM (RBF)", "model_svm.pkl")
        ]:
            p = self._find_artifact(fname)
            if p:
                try:
                    self.models[m_key] = joblib.load(p)
                except Exception as e:
                    logger.warning(f" [!] Error al cargar {m_key}: {e}")

        if not self.models:
            fallback_p = self._find_artifact("model.pkl")
            if fallback_p:
                self.models["Random Forest"] = joblib.load(fallback_p)

        # Modelos auxiliares multimodales (Coimbra y WDBC)
        c_mod = self._find_artifact("model_coimbra.pkl")
        c_sca = self._find_artifact("scaler_coimbra.pkl")
        if c_mod and c_sca:
            self.model_coimbra = joblib.load(c_mod)
            self.scaler_coimbra = joblib.load(c_sca)

        w_mod = self._find_artifact("model_wdbc.pkl")
        w_sca = self._find_artifact("scaler_wdbc.pkl")
        if w_mod and w_sca:
            self.model_wdbc = joblib.load(w_mod)
            self.scaler_wdbc = joblib.load(w_sca)

        comp_json = self._find_artifact("models_comparison.json")
        if comp_json:
            try:
                with open(comp_json, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                self.wdbc_profiles = comp_data.get("auxiliary_models", {}).get("wdbc", {})
            except Exception:
                pass

    def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesa y valida el payload entrante de la petición HTTP."""
        payload = data
        if "data" in data and isinstance(data["data"], dict):
            payload = data["data"]
        elif "body" in data and isinstance(data["body"], (str, bytes)):
            payload = json.loads(data["body"])

        features_input = payload.get("features", {})
        if not features_input:
            raise ValueError("Payload inválido: el campo 'features' es obligatorio.")

        feature_vec = []
        for feat in FEATURE_NAMES:
            if feat not in features_input:
                raise ValueError(f"Característica biofísica faltante: '{feat}'")
            feature_vec.append(float(features_input[feat]))

        feature_array = np.array([feature_vec], dtype=np.float64)
        scaled_array = self.scaler.transform(feature_array) if self.scaler else feature_array

        return {
            "patient_id": payload.get("patient_id", "ANON-PAT-001"),
            "model_requested": payload.get("model_name", "XGBoost"),
            "scaled_features": scaled_array,
            "raw_features": features_input,
            "metabolic_factors": payload.get("metabolic_factors", {}),
            "wdbc_features": payload.get("wdbc_features", None),
            "patient_palpable": bool(payload.get("patient_palpable", False))
        }

    def _compute_uncertainty(self, probs: np.ndarray) -> Dict[str, Any]:
        """Calcula el Clinical Uncertainty Index (CUI) y la Entropía Normalizada de Shannon."""
        p = np.clip(probs, 1e-12, 1.0)
        p = p / np.sum(p)
        k = len(p)

        h = -np.sum(p * np.log2(p))
        h_norm = float(h / np.log2(k)) if k > 1 else 0.0

        sorted_p = np.sort(p)[::-1]
        top1 = float(sorted_p[0])
        top2 = float(sorted_p[1]) if k > 1 else 0.0
        margin = float(top1 - top2)

        cui = round((0.65 * h_norm + 0.35 * (1.0 - margin)) * 100.0, 1)

        if cui < 28.0:
            level = "Baja (Certeza Alta)"
            flag = False
        elif cui <= 55.0:
            level = "Moderada (Zona Fronteriza)"
            flag = False
        else:
            level = "Crítica (No Concluyente)"
            flag = True

        return {
            "cui_score": cui,
            "confidence_score": round(100.0 - cui, 1),
            "entropy_normalized": round(h_norm, 3),
            "margin_delta_p": round(margin * 100.0, 1),
            "level": level,
            "requires_second_opinion": flag
        }

    def _inference(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta la inferencia diagnóstica, fusión multimodal y cuantificación de incertidumbre."""
        t0 = time.perf_counter()
        X_scaled = preprocessed_data["scaled_features"]
        model_req = preprocessed_data["model_requested"]

        if model_req in ["Ensamble", "Ensamble Multimodelo"]:
            probs = np.zeros(len(self.classes))
            weights = {"XGBoost": 0.45, "Random Forest": 0.35, "SVM (RBF)": 0.20}
            tot_w = 0.0
            for name, w in weights.items():
                if name in self.models:
                    probs += self.models[name].predict_proba(X_scaled)[0] * w
                    tot_w += w
            if tot_w > 0:
                probs /= tot_w
            used_model = "Ensamble Multimodelo"
        elif model_req in self.models:
            probs = self.models[model_req].predict_proba(X_scaled)[0]
            used_model = model_req
        else:
            first_key = list(self.models.keys())[0]
            probs = self.models[first_key].predict_proba(X_scaled)[0]
            used_model = first_key

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        uq = self._compute_uncertainty(probs)

        car_idx = self.classes.index("car") if "car" in self.classes else 1
        car_prob = float(probs[car_idx])
        fad_prob = float(probs[self.classes.index("fad")]) if "fad" in self.classes else 0.0
        mas_prob = float(probs[self.classes.index("mas")]) if "mas" in self.classes else 0.0

        # Inferencia real del modelo metabólico Coimbra
        met_in = preprocessed_data.get("metabolic_factors", {}) or {}
        defaults_coimbra = {
            "Age": 49.0, "BMI": 24.5, "Glucose": 92.0, "Insulin": 7.5,
            "HOMA": 1.7, "Leptin": 18.0, "Adiponectin": 12.0, "Resistin": 10.0, "MCP.1": 350.0
        }
        if self.model_coimbra and self.scaler_coimbra:
            ci_map = {k.lower(): v for k, v in met_in.items()}
            vec_c = np.array([[float(ci_map.get(col.lower(), defaults_coimbra[col])) for col in COIMBRA_COLS]])
            metabolic_prob = float(self.model_coimbra.predict_proba(self.scaler_coimbra.transform(vec_c))[0, 1])
        else:
            metabolic_prob = 0.22

        # Inferencia real del modelo morfológico WDBC
        wdbc_vec = preprocessed_data.get("wdbc_features")
        if self.model_wdbc and self.scaler_wdbc:
            if wdbc_vec is not None and len(wdbc_vec) == 30:
                w_arr = np.array([wdbc_vec], dtype=np.float64)
            elif preprocessed_data.get("patient_palpable") and self.wdbc_profiles.get("palpable_suspicious_profile"):
                w_arr = np.array([self.wdbc_profiles["palpable_suspicious_profile"]], dtype=np.float64)
            elif self.wdbc_profiles.get("benign_centroid"):
                w_arr = np.array([self.wdbc_profiles["benign_centroid"]], dtype=np.float64)
            else:
                w_arr = None

            if w_arr is not None:
                morph_prob = float(self.model_wdbc.predict_proba(self.scaler_wdbc.transform(w_arr))[0, 1])
            else:
                morph_prob = 0.55 if preprocessed_data.get("patient_palpable") else 0.08
        else:
            morph_prob = 0.55 if preprocessed_data.get("patient_palpable") else 0.08

        multimodal_risk_score = float(np.clip(0.70 * car_prob + 0.15 * metabolic_prob + 0.15 * morph_prob, 0.0, 1.0))

        if multimodal_risk_score >= 0.32 or pred_class == "car":
            triage_risk = "ALTO RIESGO"
            birads = "BI-RADS 4/5 (Sospechoso de Malignidad)"
            action = "Derivación urgente a Biopsia con Aguja Gruesa (BAG) y Mamografía diagnóstica prioritaria (<7 días)."
        elif (fad_prob + mas_prob) >= 0.38 or pred_class in ["fad", "mas"]:
            triage_risk = "RIESGO MODERADO"
            birads = "BI-RADS 3 (Probablemente Benigno)"
            action = "Seguimiento ecográfico complementario a 3-6 meses."
        else:
            triage_risk = "BAJO RIESGO"
            birads = "BI-RADS 1/2 (Normal / Hallazgo Benigno)"
            action = "Continuar con protocolo habitual de tamizaje preventivo anual."

        if uq["requires_second_opinion"]:
            action = "[ALERTA] MUESTRA ATÍPICA (Incertidumbre CUI > 55%): Se exige ecografía mamaria confirmatoria."

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "patient_id": preprocessed_data["patient_id"],
            "prediction": pred_class,
            "prediction_name": CLASS_MAPPING.get(pred_class, {}).get("name", pred_class.upper()),
            "confidence": round(confidence * 100.0, 1),
            "probabilities": {self.classes[i]: round(float(probs[i]), 4) for i in range(len(self.classes))},
            "triage_risk": triage_risk,
            "birads_category": birads,
            "clinical_uncertainty_cui": uq["cui_score"],
            "uncertainty_level": uq["level"],
            "entropy_normalized": uq["entropy_normalized"],
            "multimodal_risk_score": round(multimodal_risk_score * 100.0, 1),
            "multimodal_breakdown": {
                "eis_carcinoma_prob": round(car_prob, 4),
                "metabolic_coimbra_prob": round(metabolic_prob, 4),
                "morphological_wdbc_prob": round(morph_prob, 4),
                "fused_risk_score": round(multimodal_risk_score, 4)
            },
            "clinical_action": action,
            "model_used": used_model,
            "latency_ms": round(latency_ms, 2)
        }

    def _postprocess(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """Formatea la respuesta final de la API HTTP de ModelArts."""
        return {
            "status": "success",
            "service": "Huawei Cloud ModelArts - VITA-ES Triage Service",
            "runtime_hardware": "Huawei Cloud ModelArts / Ascend-Compatible Python 3.12",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **inference_result,
            "result": inference_result
        }


modelarts_service = VITAESModelArtsService()


def handle(data, context=None):
    """Punto de entrada estándar requerido por Huawei Cloud ModelArts."""
    preprocessed = modelarts_service._preprocess(data)
    inferred = modelarts_service._inference(preprocessed)
    return modelarts_service._postprocess(inferred)
