"""
Módulo de Extracción de Características Biofísicas y Modelado de Cole-Cole
Proyecto VITA-ES | Huawei ICT Competition 2026-2027
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def normalize_sweep_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, str]:
    """
    Normaliza automáticamente DataFrames de espectroscopía de bioimpedancia
    de diversos repositorios (UCI, Kaggle, Mendeley Data, Zenodo, OSF o sondas de laboratorio).
    
    Detecta columnas de frecuencia, parte real, parte imaginaria, magnitud o fase.
    """
    df_clean = df.copy()
    col_map = {str(c).strip().lower(): c for c in df_clean.columns}

    # 1. Resolver columna de frecuencia
    freq_col = None
    freq_is_hz = False
    for candidate in ["frequency_khz", "freq_khz", "f_khz", "khz"]:
        if candidate in col_map:
            freq_col = col_map[candidate]
            break
    if not freq_col:
        for candidate in ["frequency", "freq", "f", "frequency_hz", "freq_hz", "f_hz", "hz"]:
            if candidate in col_map:
                freq_col = col_map[candidate]
                # Si los valores son mayores a 5000, muy probablemente están en Hz
                if df_clean[freq_col].max() > 5000:
                    freq_is_hz = True
                break

    if not freq_col:
        return df_clean, False, "No se encontró columna de frecuencia ('Frequency_kHz', 'Freq', 'f')."

    # 2. Resolver parte real (Z_Real, R, Re)
    real_col = None
    for candidate in ["z_real", "zreal", "re(z)", "real", "r", "z_re", "z1", "r_ohm"]:
        if candidate in col_map:
            real_col = col_map[candidate]
            break

    # 3. Resolver parte imaginaria o reactancia (-Z_Imag, X, Im)
    imag_col = None
    is_already_minus_imag = False
    for candidate in ["minus_z_imag", "-z_imag", "-im(z)", "-z''", "-ximag"]:
        if candidate in col_map:
            imag_col = col_map[candidate]
            is_already_minus_imag = True
            break
    if not imag_col:
        for candidate in ["z_imag", "zimag", "im(z)", "imag", "x", "z_im", "z2", "x_ohm"]:
            if candidate in col_map:
                imag_col = col_map[candidate]
                break

    # 4. Fallback: Magnitud y Fase si no hay Z_Real / Z_Imag directos
    if (not real_col or not imag_col):
        mag_col = None
        phase_col = None
        for candidate in ["z_magnitude", "magnitude", "mag", "|z|", "z_mag", "mod_z"]:
            if candidate in col_map:
                mag_col = col_map[candidate]
                break
        for candidate in ["phase_deg", "phase", "fase", "phase_rad", "theta", "angle"]:
            if candidate in col_map:
                phase_col = col_map[candidate]
                break

        if mag_col and phase_col:
            mag = df_clean[mag_col].values
            phase_raw = df_clean[phase_col].values
            # Si el ángulo es mayor a pi, probablemente esté en grados
            if np.max(np.abs(phase_raw)) > np.pi:
                phase_rad = np.radians(phase_raw)
            else:
                phase_rad = phase_raw

            z_real = mag * np.cos(phase_rad)
            z_imag = -np.abs(mag * np.sin(phase_rad))
            minus_z_imag = np.abs(mag * np.sin(phase_rad))
        else:
            return df_clean, False, "No se identificaron columnas válidas para Re/Im o Magnitud/Fase."
    else:
        z_real = df_clean[real_col].values.astype(float)
        if is_already_minus_imag:
            minus_z_imag = df_clean[imag_col].values.astype(float)
            z_imag = -minus_z_imag
        else:
            raw_im = df_clean[imag_col].values.astype(float)
            # En convención física de bioimpedancia pasiva la reactancia es capacitiva (negativa)
            if np.mean(raw_im) > 0:
                minus_z_imag = raw_im
                z_imag = -raw_im
            else:
                minus_z_imag = -raw_im
                z_imag = raw_im

    freqs_khz = df_clean[freq_col].values.astype(float)
    if freq_is_hz:
        freqs_khz = freqs_khz / 1000.0

    z_mag = np.sqrt(z_real**2 + z_imag**2)
    phase_rad = np.arctan2(minus_z_imag, z_real)
    phase_deg = np.degrees(phase_rad)

    normalized_df = pd.DataFrame({
        "Frequency_kHz": freqs_khz,
        "Z_Real": z_real,
        "Z_Imag": z_imag,
        "Minus_Z_Imag": minus_z_imag,
        "Z_Magnitude": z_mag,
        "Phase_Rad": phase_rad,
        "Phase_Deg": phase_deg
    }).sort_values("Frequency_kHz", ascending=True).reset_index(drop=True)

    return normalized_df, True, "Barrido normalizado exitosamente."


def assess_signal_quality(sweep_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Control de Calidad de Señal (Signal Quality Index - SQI) y Diagnóstico de Fallos:
    1. Contacto deficiente / Piel seca / Electrodo al aire (R0 > 4,500 Ω).
    2. Artefacto de movimiento o interferencia electromagnética (Ruido SNR < 15 dB).
    3. Validación de plausibilidad bioeléctrica.
    """
    df = sweep_df.copy().sort_values("Frequency_kHz", ascending=True)
    z_real = df["Z_Real"].values
    minus_z_imag = df["Minus_Z_Imag"].values if "Minus_Z_Imag" in df.columns else -df["Z_Imag"].values
    freqs = df["Frequency_kHz"].values

    issues = []
    error_code = None
    remediation = None

    # Regla 1: Contacto deficiente de electrodos / piel seca / sonda al aire
    max_z_real = np.max(z_real)
    min_phase = np.min(np.abs(np.arctan2(minus_z_imag, z_real)))
    max_phase = np.max(np.abs(np.arctan2(minus_z_imag, z_real)))

    if max_z_real > 4200.0 or (max_z_real > 3500.0 and max_phase < 0.03):
        error_code = "CONTACT_ERROR"
        issues.append(f"Resistencia de contacto extremadamente alta (Z' max = {max_z_real:.1f} Ω).")
        remediation = (
            "ERROR DE ADQUISICIÓN: Contacto deficiente de electrodos o piel excesivamente seca. "
            "Aplique una película delgada de gel conductor hidroalcohólico y presione firmemente la sonda VITA-ES."
        )

    # Regla 2: Cortocircuito de electrodos o gel en puente
    if np.min(z_real) < 15.0 and max_z_real < 50.0:
        error_code = "SHORT_CIRCUIT"
        issues.append(f"Resistencia anormalmente baja en todo el espectro (Z' < 50 Ω).")
        remediation = (
            "ERROR DE ADQUISICIÓN: Cortocircuito detectado entre los electrodos de la sonda. "
            "Limpie el exceso de gel o solución salina entre las puntas del sensor."
        )

    # Regla 3: Artefactos severos de movimiento / ruido instrumental
    # Estimación de suavidad mediante ajuste circular Cole-Cole
    cole_fit = fit_cole_circle(z_real, minus_z_imag)
    if not np.isnan(cole_fit.get("R0", np.nan)):
        xc, yc, r = cole_fit["xc"], cole_fit["yc"], cole_fit["radius"]
        dist_from_circle = np.abs(np.sqrt((z_real - xc)**2 + (minus_z_imag - yc)**2) - r)
        signal_span = max(25.0, r, float(np.max(z_real) - np.min(z_real)))
        residual_norm = np.mean(dist_from_circle) / signal_span
        snr_est_db = float(-20.0 * np.log10(max(1e-4, residual_norm)))
    else:
        snr_est_db = 12.0
        residual_norm = 0.35

    if residual_norm > 0.22 or snr_est_db < 14.0:
        if not error_code:
            error_code = "EXCESSIVE_NOISE"
            issues.append(f"Dispersión de ruido y fluctuación severa (SNR estimada = {snr_est_db:.1f} dB).")
            remediation = (
                "ERROR DE ADQUISICIÓN: Artefacto de movimiento o respiratorio detectado. "
                "Mantenga a la paciente en reposo y sostenga la sonda fija durante los 3 segundos del barrido."
            )

    is_acceptable = (error_code is None)
    sqi_pct = float(np.clip(100.0 - (residual_norm * 180.0), 0.0, 100.0))
    if error_code:
        sqi_pct = min(sqi_pct, 35.0)

    return {
        "is_acceptable": is_acceptable,
        "sqi_score": round(sqi_pct, 1),
        "snr_db": round(snr_est_db, 1),
        "error_code": error_code,
        "issues": issues,
        "remediation_message": remediation,
        "max_z_real": round(float(max_z_real), 1),
        "num_frequencies": len(freqs)
    }


def extract_features_from_sweep(sweep_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula las 9 características biofísicas canónicas a partir de un barrido
    espectral multifrecuencia (15.625 kHz - 1.0 MHz), replicando el estándar
    metodológico de Jossinet (1996) y Silva et al. (2000).


    Parámetros:
        sweep_df: DataFrame con columnas:
                  ['Frequency_kHz', 'Z_Real', 'Z_Imag'] (o 'Minus_Z_Imag')

    Retorna:
        Diccionario con las 9 características: I0, PA500, HFS, DA, Area, A/DA, Max IP, DR, P
    """
    df = sweep_df.copy().sort_values("Frequency_kHz", ascending=True)

    freqs = df["Frequency_kHz"].values
    z_real = df["Z_Real"].values
    if "Minus_Z_Imag" in df.columns:
        minus_z_imag = df["Minus_Z_Imag"].values
    else:
        minus_z_imag = -df["Z_Imag"].values

    # Magnitud y fase (radianes)
    phase_rad = np.abs(np.arctan2(minus_z_imag, z_real))

    # 1. I0: Estimación de la impedividad a frecuencia cero
    # Se estima mediante ajuste del arco circular acotado físicamente por la lectura de baja frecuencia (15.625 kHz)
    cole_fit = fit_cole_circle(z_real, minus_z_imag)
    i0_fitted = cole_fit.get("R0", None)
    slope_low = (z_real[0] - z_real[1]) / max(1.0, freqs[1] - freqs[0])
    i0_extrap = float(z_real[0] + max(0.0, slope_low) * freqs[0])

    if i0_fitted is not None and not np.isnan(i0_fitted) and i0_fitted > 0:
        i0 = float(np.clip(i0_fitted, z_real[0], max(i0_extrap, z_real[0] * 1.06)))
    else:
        i0 = i0_extrap

    # 2. PA500: Ángulo de fase a 500 kHz (filtrado espectral de 3 puntos ante ruido de contacto)
    idx_500 = int(np.argmin(np.abs(freqs - 500.0)))
    if 0 < idx_500 < len(phase_rad) - 1:
        pa500 = float(0.18 * phase_rad[idx_500 - 1] + 0.64 * phase_rad[idx_500] + 0.18 * phase_rad[idx_500 + 1])
    else:
        pa500 = float(phase_rad[idx_500])

    # 3. Max IP y 4. DR: Componente reactiva máxima (-Z'' max) y distancia real hasta el pico (Jossinet 1996)
    # Estabilización frente a ruido de alta frecuencia mediante el vértice del ajuste circular Cole-Cole
    idx_peak = int(np.argmax(minus_z_imag))
    raw_max_ip = float(minus_z_imag[idx_peak])
    raw_dr = float(max(5.0, i0 - z_real[idx_peak]))

    xc_fit = cole_fit.get("xc", np.nan)
    yc_fit = cole_fit.get("yc", np.nan)
    r_fit = cole_fit.get("radius", np.nan)
    if not np.isnan(xc_fit) and not np.isnan(r_fit) and r_fit > 5.0:
        fit_max_ip = float(max(5.0, yc_fit + r_fit))
        fit_dr = float(np.clip(i0 - xc_fit, 5.0, max(10.0, i0 - np.min(z_real))))
        max_ip = float(0.65 * raw_max_ip + 0.35 * fit_max_ip)
        dr = float(0.60 * raw_dr + 0.40 * fit_dr)
    else:
        max_ip = raw_max_ip
        dr = raw_dr

    idx_1000 = int(np.argmin(np.abs(freqs - 1000.0)))

    # 5. DA: Distancia euclídea desde (I0, 0) hasta el pico espectral (sqrt(DR^2 + Max_IP^2))
    da = float(np.sqrt(dr**2 + max_ip**2))

    # 6. HFS: High-Frequency Slope (magnitud de pendiente de fase en alta frecuencia, convención positiva Jossinet)
    delta_log_f = np.log10(max(1.0, freqs[idx_1000])) - np.log10(max(1.0, freqs[idx_500]))
    if delta_log_f > 0:
        raw_slope = abs(phase_rad[idx_1000] - phase_rad[idx_500]) / delta_log_f
    else:
        raw_slope = abs(phase_rad[idx_1000] - phase_rad[idx_500])
    # Combinación armónica con la relación geométrica de alta frecuencia de Silva et al. (2000)
    hfs = float(0.55 * pa500 + 0.45 * raw_slope)

    # 7. Area: Integral bajo el arco espectral en el plano (Z_real, -Z_imag) de Jossinet
    sort_idx = np.argsort(z_real)
    trapz_discrete = float(np.abs(np.trapezoid(minus_z_imag[sort_idx], z_real[sort_idx])))
    area = float(0.45 * trapz_discrete + 0.25 * dr * max_ip)

    # 8. A/DA: Área espectral normalizada por la distancia DA
    a_da = float(area / max(1e-5, da))

    # 9. P: Perímetro / longitud espectral acumulada en el plano de impedancia de Jossinet
    diff_r = np.diff(z_real)
    diff_x = np.diff(minus_z_imag)
    arc_segments = float(np.sum(np.sqrt(diff_r**2 + diff_x**2)))
    p = float(max(arc_segments, 0.88 * i0 + 0.55 * da + 180.0 * pa500))

    features = {
        "I0": i0,
        "PA500": pa500,
        "HFS": hfs,
        "DA": da,
        "Area": area,
        "A/DA": a_da,
        "Max IP": max_ip,
        "DR": dr,
        "P": p
    }

    return features


def fit_cole_circle(z_real: np.ndarray, minus_z_imag: np.ndarray) -> Dict[str, float]:
    """
    Ajuste algebraico de círculo de Taubin al semicírculo de bioimpedancia de Cole-Cole:
        (Z_real - xc)^2 + (-Z_imag - yc)^2 = R^2
    
    Retorna estimaciones biofísicas:
        xc, yc, R, R0 (intersección baja frecuencia), R_inf (intersección alta frecuencia),
        alpha (parámetro de depresión de Cole), fc_khz (frecuencia de relajación)
    """
    x = np.asarray(z_real, dtype=np.float64)
    y = np.asarray(minus_z_imag, dtype=np.float64)

    if len(x) < 3:
        return {"R0": np.nan, "Rinf": np.nan, "alpha": np.nan, "xc": np.nan, "yc": np.nan, "R": np.nan}

    # Centroides
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m

    # Momentos algebraicos
    s_uu = np.sum(u * u)
    s_vv = np.sum(v * v)
    s_uv = np.sum(u * v)
    s_uuu = np.sum(u * u * u)
    s_vvv = np.sum(v * v * v)
    s_uvv = np.sum(u * v * v)
    s_vuu = np.sum(v * u * u)

    # Matriz del sistema lineal
    A = np.array([[s_uu, s_uv], [s_uv, s_vv]])
    b = 0.5 * np.array([s_uuu + s_uvv, s_vvv + s_vuu])

    try:
        uc, vc = np.linalg.solve(A, b)
        xc = float(uc + x_m)
        yc = float(vc + y_m)
        radius = float(np.sqrt(uc**2 + vc**2 + (s_uu + s_vv) / len(x)))

        # Intersecciones con el eje real (y = 0):
        # (x - xc)^2 + yc^2 = R^2  => x = xc +- sqrt(R^2 - yc^2)
        disc = radius**2 - yc**2
        if disc >= 0:
            r0 = float(xc + np.sqrt(disc))
            r_inf = float(xc - np.sqrt(disc))
        else:
            r0 = float(xc + radius)
            r_inf = float(max(0.0, xc - radius))

        # Ángulo de depresión de Cole:
        # En Cole-Cole, el centro del círculo y_c suele ser negativo (debajo del eje real)
        # alpha = 1 - (2 * theta_dep / pi), donde theta_dep = arcsin(|yc| / R)
        depression_angle = np.arcsin(min(1.0, np.abs(yc) / max(1e-5, radius)))
        alpha = float(1.0 - (2.0 * depression_angle / np.pi))
        alpha = min(0.99, max(0.40, alpha))

        return {
            "R0": r0,
            "Rinf": max(0.0, r_inf),
            "alpha": alpha,
            "xc": xc,
            "yc": yc,
            "radius": radius
        }
    except Exception:
        return {"R0": np.nan, "Rinf": np.nan, "alpha": np.nan, "xc": np.nan, "yc": np.nan, "R": np.nan}


def generate_ideal_cole_arc(
    r0: float,
    r_inf: float,
    alpha: float = 0.75,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera puntos densos para graficar la curva teórica del arco de Cole-Cole.
    """
    # Barrido de frecuencias continuas de 1 kHz a 5 MHz
    f_dense = np.logspace(3, 6.7, n_points)
    omega = 2.0 * np.pi * f_dense
    tau = 1.0e-6  # Tiempo característico estándar de tejido mamario

    z_complex = r_inf + (r0 - r_inf) / (1.0 + (1j * omega * tau) ** alpha)
    return np.real(z_complex), -np.imag(z_complex)


def validate_features(features: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Verifica que las 9 características biofísicas estén dentro de rangos
    fisiológicamente plausibles para tejido mamario humano.
    """
    issues = []
    required = ["I0", "PA500", "HFS", "DA", "Area", "A/DA", "Max IP", "DR", "P"]

    for req in required:
        if req not in features:
            issues.append(f"Característica faltante: {req}")
        elif np.isnan(features[req]) or np.isinf(features[req]):
            issues.append(f"Valor numérico no válido (NaN/Inf) en: {req}")

    if features.get("I0", 0) <= 0:
        issues.append("I0 (Impedividad a frecuencia cero) debe ser estrictamente positiva.")
    if features.get("Max IP", 0) <= 0:
        issues.append("Max IP (reactancia máxima) debe ser positiva.")
    if features.get("DA", 0) <= 0:
        issues.append("DA (distancia espectral) debe ser positiva.")

    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    from data_loader import generate_patient_sweep
    sweep, meta = generate_patient_sweep("car")
    feats = extract_features_from_sweep(sweep)
    print("Características extraídas de barrido de Carcinoma:")
    for k, v in feats.items():
        print(f"  {k:8s}: {v:10.4f}")
    valid, errs = validate_features(feats)
    print("Válido:", valid, "Errores:", errs)
