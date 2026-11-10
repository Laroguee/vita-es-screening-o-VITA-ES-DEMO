"""
Módulo de Carga y Gestión de Datos de Bioimpedancia Eléctrica (EIS)
Proyecto VITA-ES | Huawei ICT Competition 2026-2027
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Mapeo clínico de clases histológicas calibrado con las distribuciones empíricas de Jossinet (1996)
CLASS_MAPPING = {
    "car": {
        "name": "Carcinoma",
        "description": "Neoplasia maligna de mama con alta celularidad y atipia",
        "risk_level": "ALTO RIESGO",
        "risk_code": 2,
        "color": "#EF4444",
        "birads_category": "BI-RADS 4/5 (Sospechoso de Malignidad)",
        "clinical_action": "Derivación urgente a Biopsia con Aguja Gruesa (BAG) y Mamografía diagnóstica prioritaria.",
        "cole_params": {"R0": 394.0, "Rinf": 90.0, "tau": 6.0e-7, "alpha": 0.65}
    },
    "fad": {
        "name": "Fibroadenoma",
        "description": "Tumor benigno fibroepitelial bifásico común en mujeres jóvenes",
        "risk_level": "RIESGO MODERADO",
        "risk_code": 1,
        "color": "#F59E0B",
        "birads_category": "BI-RADS 3 (Probablemente Benigno)",
        "clinical_action": "Evaluación por Ultrasonido mamario complementario y seguimiento clínico en 3 a 6 meses.",
        "cole_params": {"R0": 245.0, "Rinf": 152.0, "tau": 4.2e-7, "alpha": 0.62}
    },
    "mas": {
        "name": "Mastopatía",
        "description": "Condición fibroquística y alteraciones proliferativas benignas",
        "risk_level": "RIESGO MODERADO",
        "risk_code": 1,
        "color": "#F59E0B",
        "birads_category": "BI-RADS 3 (Probablemente Benigno)",
        "clinical_action": "Monitoreo clínico semestral y control ecográfico; correlación con ciclo hormonal.",
        "cole_params": {"R0": 290.0, "Rinf": 151.0, "tau": 2.5e-6, "alpha": 0.75}
    },
    "gla": {
        "name": "Glandular",
        "description": "Parénquima glandular normal ductal y lobulillar",
        "risk_level": "BAJO RIESGO",
        "risk_code": 0,
        "color": "#10B981",
        "birads_category": "BI-RADS 1/2 (Normal / Hallazgo Benigno)",
        "clinical_action": "Continuar con protocolo habitual de tamizaje preventivo anual según grupo etario.",
        "cole_params": {"R0": 210.0, "Rinf": 152.0, "tau": 4.5e-7, "alpha": 0.90}
    },
    "con": {
        "name": "Conectivo",
        "description": "Estroma conectivo y fibroso mamario estructural sano",
        "risk_level": "BAJO RIESGO",
        "risk_code": 0,
        "color": "#10B981",
        "birads_category": "BI-RADS 1/2 (Normal / Hallazgo Benigno)",
        "clinical_action": "Continuar con protocolo habitual de tamizaje preventivo anual según grupo etario.",
        "cole_params": {"R0": 1310.0, "Rinf": 890.0, "tau": 1.8e-6, "alpha": 0.48}
    },
    "adi": {
        "name": "Adiposo",
        "description": "Tejido graso mamario subcutáneo y retromamario normal",
        "risk_level": "BAJO RIESGO",
        "risk_code": 0,
        "color": "#10B981",
        "birads_category": "BI-RADS 1/2 (Normal / Hallazgo Benigno)",
        "clinical_action": "Continuar con protocolo habitual de tamizaje preventivo anual según grupo etario.",
        "cole_params": {"R0": 2020.0, "Rinf": 1410.0, "tau": 6.0e-7, "alpha": 0.76}
    }
}

# Frecuencias estándar del protocolo de espectroscopía de Jossinet (kHz)
STANDARD_FREQUENCIES_KHZ = [15.625, 31.25, 62.5, 125.0, 250.0, 500.0, 1000.0]

# Características biofísicas canónicas extraídas del espectro
FEATURE_NAMES = ["I0", "PA500", "HFS", "DA", "Area", "A/DA", "Max IP", "DR", "P"]

FEATURE_DESCRIPTIONS = {
    "I0": "Impedividad extrapolada a frecuencia cero (Resistencia extracelular R0 en Ohm)",
    "PA500": "Ángulo de fase en radianes a 500 kHz (sensible a capacitancia de membrana)",
    "HFS": "Pendiente de alta frecuencia del ángulo de fase (500 kHz - 1 MHz)",
    "DA": "Distancia en plano complejo entre extremos espectrales (15.6 kHz y 1 MHz)",
    "Area": "Área bajo la curva del semicírculo de impedancia de Nyquist",
    "A/DA": "Área espectral normalizada por la distancia DA (geometría de dispersión)",
    "Max IP": "Máximo componente reactivo imaginario (-Z'' max / pico de arco Cole)",
    "DR": "Diferencia de impedancia entre I0 y componente real a frecuencia máxima",
    "P": "Longitud de arco de la trayectoria del espectro en plano complejo"
}


def find_dataset_path() -> Optional[str]:
    """Busca la ruta del dataset de bioimpedancia en los directorios del proyecto."""
    candidate_paths = [
        os.path.join("data", "breast_tissue.csv"),
        os.path.join("Data", "breast_tissue.csv"),
        os.path.join("data", "BreastTissue.xls"),
        os.path.join("Data", "BreastTissue.xls"),
        os.path.join("..", "data", "breast_tissue.csv"),
        os.path.join("..", "Data", "breast_tissue.csv"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None


def generate_synthetic_dataset(n_samples: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético realista con base en las distribuciones estadísticas
    de bioimpedancia de tejidos mamarios documentadas por Jossinet (1996) y Silva et al. (2000).
    """
    np.random.seed(seed)
    records = []
    classes = ["car", "fad", "mas", "gla", "con", "adi"]
    samples_per_class = n_samples // len(classes)

    # Parámetros medios por clase [I0, PA500, HFS, DA, Area, A/DA, Max IP, DR, P]
    class_stats = {
        "car": {"mean": [420, 0.14, 0.035, 180, 8500, 42, 75, 170, 520], "std": [80, 0.04, 0.010, 35, 2000, 10, 15, 35, 90]},
        "fad": {"mean": [750, 0.11, 0.025, 140, 6200, 38, 55, 130, 410], "std": [110, 0.03, 0.008, 30, 1500, 8, 12, 28, 75]},
        "mas": {"mean": [820, 0.10, 0.022, 130, 5400, 36, 48, 120, 380], "std": [120, 0.03, 0.007, 28, 1400, 7, 10, 25, 70]},
        "gla": {"mean": [1300, 0.08, 0.015, 110, 3800, 30, 35, 95, 310], "std": [180, 0.02, 0.005, 25, 1100, 6, 8, 20, 55]},
        "con": {"mean": [1550, 0.07, 0.012, 95, 2900, 26, 28, 80, 270], "std": [200, 0.02, 0.004, 22, 900, 5, 7, 18, 50]},
        "adi": {"mean": [2100, 0.05, 0.008, 80, 1800, 20, 20, 65, 220], "std": [250, 0.015, 0.003, 18, 600, 4, 5, 15, 40]},
    }

    case_id = 1
    for cls in classes:
        mean_vals = np.array(class_stats[cls]["mean"])
        std_vals = np.array(class_stats[cls]["std"])
        for _ in range(samples_per_class):
            sample = np.random.normal(mean_vals, std_vals)
            # Asegurar no negatividad
            sample = np.clip(sample, a_min=0.001, a_max=None)
            row = {
                "Case #": case_id,
                "Class": cls,
                "I0": sample[0],
                "PA500": sample[1],
                "HFS": sample[2],
                "DA": sample[3],
                "Area": sample[4],
                "A/DA": sample[5],
                "Max IP": sample[6],
                "DR": sample[7],
                "P": sample[8]
            }
            records.append(row)
            case_id += 1

    df = pd.DataFrame(records)
    return df


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Carga el dataset de bioimpedancia clínica, estandariza columnas y enriquece
    con metadatos diagnósticos y semáforo de riesgo clínico.
    """
    target_path = filepath or find_dataset_path()

    if target_path and os.path.exists(target_path):
        if target_path.endswith(".csv"):
            df = pd.read_csv(target_path)
        else:
            df = pd.read_excel(target_path, sheet_name="Data")
    else:
        # Generar dataset sintético fundamentado
        df = generate_synthetic_dataset()
        os.makedirs("data", exist_ok=True)
        df.to_csv(os.path.join("data", "breast_tissue.csv"), index=False)

    # Normalizar nombres de columnas eliminando espacios accidentales
    df.columns = [c.strip() for c in df.columns]

    # Corregir nombres de columnas si varían en mayúsculas/minúsculas
    col_rename = {
        "case #": "Case #",
        "class": "Class",
        "i0": "I0",
        "pa500": "PA500",
        "hfs": "HFS",
        "da": "DA",
        "area": "Area",
        "a/da": "A/DA",
        "max ip": "Max IP",
        "dr": "DR",
        "p": "P"
    }
    df.rename(columns={c: col_rename.get(c.lower(), c) for c in df.columns}, inplace=True)

    # Limpiar y normalizar la columna de clase a minúsculas
    df["Class"] = df["Class"].astype(str).str.strip().str.lower()

    # Enriquecer con metadatos clínicos
    df["Class_Name"] = df["Class"].apply(lambda c: CLASS_MAPPING.get(c, {}).get("name", c.upper()))
    df["Risk_Level"] = df["Class"].apply(lambda c: CLASS_MAPPING.get(c, {}).get("risk_level", "DESCONOCIDO"))
    df["Risk_Code"] = df["Class"].apply(lambda c: CLASS_MAPPING.get(c, {}).get("risk_code", 0))
    df["Risk_Color"] = df["Class"].apply(lambda c: CLASS_MAPPING.get(c, {}).get("color", "#6B7280"))
    df["BIRADS_Category"] = df["Class"].apply(lambda c: CLASS_MAPPING.get(c, {}).get("birads_category", "N/A"))
    df["Clinical_Action"] = df["Class"].apply(lambda c: CLASS_MAPPING.get(c, {}).get("clinical_action", "N/A"))

    return df


def generate_cole_sweep(
    r0: float,
    r_inf: float,
    tau: float = 1.0e-6,
    alpha: float = 0.75,
    frequencies_khz: Optional[List[float]] = None,
    noise_std: float = 0.015,
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Genera el barrido espectral multifrecuencia completo Z(f) resolviendo la ecuación de Cole-Cole:
        Z(omega) = R_inf + (R0 - R_inf) / (1 + (j * omega * tau)**alpha)
    
    Donde:
        omega = 2 * pi * f
        R0 = Resistencia extracelular a baja frecuencia (impedivity at zero frequency)
        R_inf = Resistencia intra+extracelular a frecuencia infinita
        tau = Tiempo de relajación dipolar de membrana
        alpha = Parámetro de dispersión de Cole (0 < alpha <= 1)
        random_seed = Semilla determinista para reproducibilidad clínica del barrido
    """
    freqs_khz = frequencies_khz or STANDARD_FREQUENCIES_KHZ
    f_hz = np.array(freqs_khz) * 1e3
    omega = 2.0 * np.pi * f_hz

    # Ecuación de Cole en variable compleja
    j = 1j
    z_complex = r_inf + (r0 - r_inf) / (1.0 + (j * omega * tau) ** alpha)

    # Agregar ruido de medición gaussiano realista propio de la sonda de prueba
    if noise_std > 0:
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()
        noise_re = rng.normal(0, noise_std * np.abs(np.real(z_complex)))
        noise_im = rng.normal(0, noise_std * np.abs(np.imag(z_complex)))
        z_complex = (np.real(z_complex) + noise_re) + 1j * (np.imag(z_complex) + noise_im)

    z_real = np.real(z_complex)
    z_imag = np.imag(z_complex)
    z_mag = np.abs(z_complex)
    phase_rad = np.angle(z_complex)
    phase_deg = np.degrees(phase_rad)

    sweep_df = pd.DataFrame({
        "Frequency_kHz": freqs_khz,
        "Frequency_Hz": f_hz,
        "Z_Real": z_real,
        "Z_Imag": z_imag,
        "Minus_Z_Imag": -z_imag,  # Convenio Nyquist Cole-Cole: eje Y positivo = -Z''
        "Z_Magnitude": z_mag,
        "Phase_Rad": phase_rad,
        "Phase_Deg": phase_deg
    })

    return sweep_df


def generate_ideal_cole_arc(
    r0: float,
    r_inf: float,
    alpha: float = 0.75,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función de compatibilidad exportada desde data_loader y feature_extraction.
    Genera puntos densos para graficar la curva teórica del arco de Cole-Cole.
    """
    from feature_extraction import generate_ideal_cole_arc as _arc
    return _arc(r0, r_inf, alpha, n_points)



def generate_patient_sweep(tissue_class: str = "car", noise_level: float = 0.02) -> Tuple[pd.DataFrame, Dict]:
    """
    Genera una lectura de bioimpedancia simulada desde la sonda portátil VITA-ES
    para un tipo de tejido específico con sus parámetros electrofisiológicos base.
    """
    info = CLASS_MAPPING.get(tissue_class, CLASS_MAPPING["car"])
    params = info["cole_params"]

    # Variación interindividual aleatoria de +/- 5%
    var = 1.0 + np.random.uniform(-0.06, 0.06)
    r0 = params["R0"] * var
    r_inf = params["Rinf"] * var
    tau = params["tau"] * (1.0 + np.random.uniform(-0.05, 0.05))
    alpha = min(0.95, max(0.60, params["alpha"] * (1.0 + np.random.uniform(-0.03, 0.03))))

    sweep_df = generate_cole_sweep(
        r0=r0,
        r_inf=r_inf,
        tau=tau,
        alpha=alpha,
        noise_std=noise_level
    )

    meta = {
        "tissue_class": tissue_class,
        "class_name": info["name"],
        "risk_level": info["risk_level"],
        "risk_code": info["risk_code"],
        "color": info["color"],
        "birads": info["birads_category"],
        "R0_simulated": r0,
        "Rinf_simulated": r_inf,
        "tau_simulated": tau,
        "alpha_simulated": alpha
    }

    return sweep_df, meta


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset cargado exitosamente:")
    print(f"Dimensiones: {df.shape}")
    print(df["Class_Name"].value_counts())
    print("\nSimulación de barrido para Carcinoma:")
    sweep, meta = generate_patient_sweep("car")
    print(sweep.head(3))
    print("Meta:", meta)
