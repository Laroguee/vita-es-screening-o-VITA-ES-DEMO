"""
VITA-ES: Plataforma de Tamizaje de Cáncer de Mama mediante Espectroscopía de Bioimpedancia Eléctrica (EIS)
Huawei ICT Competition 2026-2027 - Innovation Track
"""

__version__ = "1.0.0"
__author__ = "VITA-ES Biomedical AI Engineering Team"

from .data_loader import (
    CLASS_MAPPING,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    STANDARD_FREQUENCIES_KHZ,
    generate_cole_sweep,
    generate_patient_sweep,
    load_dataset,
)
from .feature_extraction import (
    assess_signal_quality,
    extract_features_from_sweep,
    fit_cole_circle,
    generate_ideal_cole_arc,
    normalize_sweep_dataframe,
    validate_features,
)
from .inference import EISInferenceEngine

__all__ = [
    "CLASS_MAPPING",
    "FEATURE_DESCRIPTIONS",
    "FEATURE_NAMES",
    "STANDARD_FREQUENCIES_KHZ",
    "generate_cole_sweep",
    "generate_patient_sweep",
    "load_dataset",
    "assess_signal_quality",
    "extract_features_from_sweep",
    "fit_cole_circle",
    "generate_ideal_cole_arc",
    "normalize_sweep_dataframe",
    "validate_features",
    "EISInferenceEngine",
]
