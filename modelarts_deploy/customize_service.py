"""
Huawei Cloud ModelArts Custom Service Script
Proyecto VITA-ES | Huawei ICT Competition 2026-2027

Script oficial de servicio de inferencia para Huawei ModelArts Online Service.
Maneja el ciclo de vida del microservicio de IA:
- _preprocess(): Valida payload JSON de bioimpedancia y datos clínicos
- _inference(): Ejecuta inferencia multimodelo y cálculo de incertidumbre (CUI)
- _postprocess(): Devuelve respuesta JSON con semáforo BI-RADS y recomendaciones
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import joblib
import numpy as np

# Configurar logger para ModelArts
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VITA-ES-ModelArts")

FEATURE_NAMES = ["I0", "PA500", "HFS", "DA", "Area", "A/DA", "Max IP", "DR", "P"]

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
        self.classes = ["adi", "car", "con", "fad", "gla", "mas"]
        self._load_models()

    def _load_models(self):
        """Carga artefactos serializados disponibles en el directorio del servicio."""
        logger.info(f"[*] Cargando artefactos de ModelArts desde: {self.model_path}")
        
        # 1. Cargar Scaler
        scaler_file = os.path.join(self.model_path, "scaler.pkl")
        if not os.path.exists(scaler_file):
            scaler_file = os.path.join(self.model_path, "..", "models", "scaler.pkl")
        if os.path.exists(scaler_file):
            self.scaler = joblib.load(scaler_file)
            logger.info(" [+] Scaler estándar cargado exitosamente.")

        # 2. Cargar Modelos (XGBoost, Random Forest, SVM)
        for m_key, fname in [("XGBoost", "model_xgb.pkl"), ("Random Forest", "model_rf.pkl"), ("SVM (RBF)", "model_svm.pkl")]:
            p = os.path.join(self.model_path, fname)
            if not os.path.exists(p):
                p = os.path.join(self.model_path, "..", "models", fname)
            if os.path.exists(p):
                try:
                    self.models[m_key] = joblib.load(p)
                    logger.info(f" [+] Modelo {m_key} cargado exitosamente.")
                except Exception as e:
                    logger.warning(f" [!] Error al cargar {m_key}: {e}")

        # Fallback si no hay modelo específico
        if not self.models:
            fallback_p = os.path.join(self.model_path, "model.pkl")
            if not os.path.exists(fallback_p):
                fallback_p = os.path.join(self.model_path, "..", "models", "model.pkl")
            if os.path.exists(fallback_p):
                self.models["Random Forest"] = joblib.load(fallback_p)

    def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesa y valida el payload entrante de la petición HTTP.
        """
        # ModelArts puede enviar data como diccionario directo o bajo clave 'data' / 'body'
        payload = data
        if "data" in data and isinstance(data["data"], dict):
            payload = data["data"]
        elif "body" in data and isinstance(data["body"], (str, bytes)):
            payload = json.loads(data["body"])

        features_input = payload.get("features", {})
        if not features_input:
            raise ValueError("Payload inválido: el campo 'features' es obligatorio.")

        # Extraer vector ordenado de 9 características
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
            "metabolic_factors": payload.get("metabolic_factors", {})
        }

    def _compute_uncertainty(self, probs: np.ndarray) -> Dict[str, Any]:
        """Calcula el Clinical Uncertainty Index (CUI) y la Entropía Normalizada de Shannon."""
        p = np.clip(probs, 1e-12, 1.0)
        p = p / np.sum(p)
        k = len(p)
        
        # Entropía de Shannon normalizada
        h = -np.sum(p * np.log2(p))
        h_norm = float(h / np.log2(k)) if k > 1 else 0.0
        
        # Margen Top-1 vs Top-2
        sorted_p = np.sort(p)[::-1]
        top1 = float(sorted_p[0])
        top2 = float(sorted_p[1]) if k > 1 else 0.0
        margin = float(top1 - top2)
        
        # CUI Score
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
        """
        Ejecuta la inferencia diagnóstica con el modelo solicitado.
        """
        t0 = time.perf_counter()
        X_scaled = preprocessed_data["scaled_features"]
        model_req = preprocessed_data["model_requested"]

        # Seleccionar modelo o Ensamble
        if model_req == "Ensamble" or model_req == "Ensamble Multimodelo":
            probs = np.zeros(len(self.classes))
            weights = {"XGBoost": 0.40, "Random Forest": 0.35, "SVM (RBF)": 0.25}
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
            # Fallback al primer modelo disponible
            first_key = list(self.models.keys())[0]
            probs = self.models[first_key].predict_proba(X_scaled)[0]
            used_model = first_key

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # Cuantificación de incertidumbre
        uq = self._compute_uncertainty(probs)

        # Regla de triaje oncológico
        car_idx = self.classes.index("car") if "car" in self.classes else 1
        car_prob = float(probs[car_idx])

        if car_prob >= 0.32 or pred_class == "car":
            triage_risk = "ALTO RIESGO"
            birads = "BI-RADS 4/5 (Sospechoso de Malignidad)"
            action = "Derivación urgente a Biopsia con Aguja Gruesa (BAG) y Mamografía diagnóstica prioritaria."
        elif pred_class in ["fad", "mas"]:
            triage_risk = "RIESGO MODERADO"
            birads = "BI-RADS 3 (Probablemente Benigno)"
            action = "Seguimiento ecográfico complementario a 3-6 meses."
        else:
            triage_risk = "BAJO RIESGO"
            birads = "BI-RADS 1/2 (Normal / Hallazgo Benigno)"
            action = "Continuar con protocolo habitual de tamizaje preventivo anual."

        if uq["requires_second_opinion"]:
            action = "[ALERTA] MUESTRA ATIPICA (Incertidumbre CUI > 55%): Se exige ecografia mamaria confirmatoria."

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
            "clinical_action": action,
            "model_used": used_model,
            "latency_ms": round(latency_ms, 2)
        }

    def _postprocess(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formatea la respuesta final de la API HTTP de ModelArts.
        """
        return {
            "status": "success",
            "service": "Huawei Cloud ModelArts - VITA-ES Triage Service",
            "runtime_hardware": "Huawei Ascend 910 Pro NPU",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": inference_result
        }


# Instancia para invocación directa en ModelArts
modelarts_service = VITAESModelArtsService()

def handle(data, context=None):
    """Punto de entrada estándar requerido por Huawei Cloud ModelArts."""
    preprocessed = modelarts_service._preprocess(data)
    inferred = modelarts_service._inference(preprocessed)
    return modelarts_service._postprocess(inferred)
