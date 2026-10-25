"""
Módulo de Cuantificación de Incertidumbre Diagnóstica Clínica (UQ)
Proyecto VITA-ES | Huawei ICT Competition 2026-2027

Proporciona métricas de confianza e incertidumbre probabilística para evitar
falsos negativos y alertar sobre casos atípicos o muestras tisulares fronterizas.
"""

from typing import Dict, List, Optional
import numpy as np


def compute_uncertainty_metrics(
    probabilities: np.ndarray,
    class_names: Optional[List[str]] = None,
    risk_level_probs: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Calcula métricas matemáticas rigurosas de incertidumbre para una predicción clínica:
    
    1. Entropía de Shannon Normalizada (H_norm):
       Mide la dispersión o falta de concentración de la distribución de probabilidad.
       H = - sum(p_k * log2(p_k))
       H_norm = H / log2(K)  [Rango: 0.0 (certeza absoluta) a 1.0 (máxima ambigüedad)]
       
    2. Margen de Confianza Top-1 vs Top-2 (Confidence Margin):
       Mide la separación entre la hipótesis diagnóstica líder y la segunda alternativa.
       Delta_P = P_(1) - P_(2)  [Rango: 0.0 (empate total) a 1.0 (dominancia absoluta)]
       
    3. Clinical Uncertainty Index (CUI):
       Métrica compuesta de seguridad médica ponderada:
       CUI = (0.65 * H_norm + 0.35 * (1.0 - Delta_P)) * 100%
       
    4. Categorización Clínica:
       - CUI < 28%: Certeza Diagnóstica Alta (Procedimiento estándar de triaje)
       - 28% <= CUI <= 55%: Incertidumbre Moderada (Patrón tisular fronterizo)
       - CUI > 55%: Incertidumbre Crítica / Caso Atípico (Alerta médica de seguridad)
    """
    probs = np.array(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / np.sum(probs)  # Normalizar a 1.0 estricto
    
    k = len(probs)
    
    # 1. Entropía de Shannon
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(k) if k > 1 else 1.0
    h_norm = float(entropy / max_entropy)
    h_norm = max(0.0, min(1.0, h_norm))
    
    # 2. Margen Top-1 vs Top-2
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if k > 1 else 0.0
    delta_p = float(top1 - top2)
    delta_p = max(0.0, min(1.0, delta_p))
    
    # 3. Índice de Incertidumbre Clínica (CUI)
    cui_score = (0.65 * h_norm + 0.35 * (1.0 - delta_p)) * 100.0
    cui_score = round(max(0.0, min(100.0, cui_score)), 1)
    
    # 4. Confianza Diagnóstica Global (Complemento de CUI)
    confidence_score = round(100.0 - cui_score, 1)
    
    # 5. Determinación de nivel de incertidumbre y acción médica
    if cui_score < 28.0:
        level = "Baja (Certeza Alta)"
        status_code = "CERTEZA_ALTA"
        badge_color = "#059669"  # Esmeralda clínico
        badge_bg = "#ECFDF5"
        clinical_interpretation = (
            "Distribución de probabilidad unimodal y altamente consistente con "
            "la firma bioeléctrica estándar. Dictamen diagnóstico con alta fiabilidad."
        )
        requires_second_opinion = False
    elif cui_score <= 55.0:
        level = "Moderada (Zona Fronteriza)"
        status_code = "INCERTIDUMBRE_MODERADA"
        badge_color = "#D97706"  # Ámbar clínico
        badge_bg = "#FFFBEB"
        clinical_interpretation = (
            "Patrón espectral en transición o superposición tisular moderada (ej. cambios "
            "fibroquísticos o tejido glandular denso). Se aconseja correlación clínica "
            "y control en 6 meses."
        )
        requires_second_opinion = False
    else:
        level = "Crítica (No Concluyente)"
        status_code = "INCERTIDUMBRE_CRITICA"
        badge_color = "#DC2626"  # Carmesí clínico
        badge_bg = "#FEF2F2"
        clinical_interpretation = (
            "ALERTA CLÍNICA: La firma de bioimpedancia presenta características atípicas "
            "o equidistancia entre clases histológicas. El algoritmo se abstiene de un descarte "
            "fácil y recomienda derivación prioritaria a ecografía mamaria / biopsia."
        )
        requires_second_opinion = True
        
    return {
        "cui_score": cui_score,
        "confidence_score": confidence_score,
        "entropy_shannon": round(entropy, 3),
        "entropy_normalized": round(h_norm, 3),
        "top1_prob": round(top1 * 100.0, 1),
        "top2_prob": round(top2 * 100.0, 1),
        "margin_delta_p": round(delta_p * 100.0, 1),
        "level": level,
        "status_code": status_code,
        "badge_color": badge_color,
        "badge_bg": badge_bg,
        "clinical_interpretation": clinical_interpretation,
        "requires_second_opinion": requires_second_opinion
    }
