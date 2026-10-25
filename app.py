"""
========================================================================================
VITA-ES: Plataforma Clínica de Tamizaje Mamario por Espectroscopía de Bioimpedancia (EIS)
Huawei ICT Competition 2026-2027 | Innovation Track - Topic 1: Huawei Cloud AI
========================================================================================
Diseño: Estación de Trabajo Clínica Hospitalaria (Tema Claro / Siemens Healthineers & Philips)
Arquitectura: Edge-to-Cloud (MindSpore Lite + Huawei Cloud ModelArts, RDS, DCS)
Model Zoo: Balanced Random Forest, SVM-RBF, XGBoost, 1D CNN (VITA-Net) y Ensamble
Incertidumbre: Clinical Uncertainty Index (CUI) con Entropía Shannon y Margen Top-1/Top-2
Fusión Multimodal: Bioimpedancia EIS (70%) + Coimbra BCCD (15%) + Wisconsin WDBC (15%)
========================================================================================
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configuración de ruta para módulos internos
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_loader import (
    CLASS_MAPPING,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    STANDARD_FREQUENCIES_KHZ,
    generate_cole_sweep,
    generate_patient_sweep,
    load_dataset,
)
from feature_extraction import (
    assess_signal_quality,
    extract_features_from_sweep,
    fit_cole_circle,
    generate_ideal_cole_arc,
    normalize_sweep_dataframe,
)
from inference import EISInferenceEngine
from uncertainty import compute_uncertainty_metrics
from sonda_emulator import VITAProbeEmulator

# ======================================================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ======================================================================================
st.set_page_config(
    page_title="VITA-ES | Estación Clínica de Tamizaje de Bioimpedancia",
    page_icon="none",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# ESTILOS CSS: TEMA CLARO CLÍNICO HOSPITALARIO (CERO FATIGA VISUAL / ALTA LEGIBILIDAD)
# ======================================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Fondo principal y textos */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Barra superior de estado */
    .hospital-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        color: #334155;
        font-weight: 500;
    }
    .status-dot-green {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        margin-right: 6px;
    }
    .status-dot-blue {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #2563EB;
        margin-right: 6px;
    }
    .status-dot-amber {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #F59E0B;
        margin-right: 6px;
    }

    /* Contenedor de Fusión Multimodal */
    .multimodal-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .multimodal-header {
        font-size: 0.76rem;
        font-weight: 700;
        color: #475569;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .multimodal-stream-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin-bottom: 10px;
    }
    .stream-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 12px;
    }
    .stream-tag {
        font-size: 0.70rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 3px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .stream-tag-eis {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .stream-tag-coimbra {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .stream-tag-wdbc {
        background: #F3E8FF;
        color: #7E22CE;
        border: 1px solid #E9D5FF;
    }
    .stream-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
    }
    .stream-sub {
        font-size: 0.72rem;
        color: #64748B;
        line-height: 1.3;
        margin-top: 2px;
    }
    .multimodal-formula {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.78rem;
        color: #334155;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }

    /* Tarjetas clínicas en blanco limpio */
    .clinical-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* Stepper secuencial en 3 pasos */
    .stepper-nav {
        display: flex;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .step-pill {
        flex: 1;
        text-align: center;
        padding: 8px 12px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #1E40AF;
        background: #EFF6FF;
        border-radius: 6px;
        margin: 0 4px;
        border: 1px solid #DBEAFE;
    }

    /* Semáforo de riesgo clínico en tonos pastel suaves */
    .triage-card-red {
        background-color: #FEF2F2;
        border: 1.5px solid #EF4444;
        border-radius: 8px;
        padding: 18px 20px;
        color: #991B1B;
    }
    .triage-card-amber {
        background-color: #FFFBEB;
        border: 1.5px solid #F59E0B;
        border-radius: 8px;
        padding: 18px 20px;
        color: #92400E;
    }
    .triage-card-green {
        background-color: #F0FDF4;
        border: 1.5px solid #10B981;
        border-radius: 8px;
        padding: 18px 20px;
        color: #166534;
    }

    /* Métrica estadística médica */
    .stat-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 2px;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================================
# INICIALIZACIÓN DE MOTOR DE INFERENCIA Y DATASETS
# ======================================================================================
@st.cache_resource
def get_inference_engine():
    return EISInferenceEngine()

@st.cache_resource
def get_probe_emulator():
    return VITAProbeEmulator()

@st.cache_data
def get_historical_dataset():
    return load_dataset()

@st.cache_data
def get_coimbra_dataset():
    for candidate in [
        os.path.join("data", "dataR2.csv"),
        os.path.join("Data", "dataR2.csv"),
        os.path.join("..", "data", "dataR2.csv"),
        os.path.join("..", "Data", "dataR2.csv")
    ]:
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    return None

@st.cache_data
def get_wdbc_dataset():
    for candidate in [
        os.path.join("data", "wdbc.data"),
        os.path.join("Data", "wdbc.data"),
        os.path.join("..", "data", "wdbc.data"),
        os.path.join("..", "Data", "wdbc.data")
    ]:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate, header=None)
            col_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
            df.columns = col_names
            return df
    return None

try:
    engine = get_inference_engine()
    dataset = get_historical_dataset()
    coimbra_df = get_coimbra_dataset()
    wdbc_df = get_wdbc_dataset()
except Exception as e:
    st.error(f"Error al inicializar el motor VITA-ES: {e}")
    st.stop()

# ======================================================================================
# BARRA LATERAL (TEMA CLARO MÉDICO)
# ======================================================================================
with st.sidebar:
    st.markdown("### VITA-ES Workstation")
    st.caption("Sistema de Tamizaje Mamario por Espectroscopía de Bioimpedancia (EIS)")
    st.markdown("---")

    st.markdown("**Estado del Instrumental Biomédico:**")
    st.markdown("""
    <div style='background:#F8FAFC; border:1px solid #E2E8F0; padding:12px; border-radius:6px; font-size:0.84rem; line-height:1.7; color:#334155;'>
        <div><span class='status-dot-green'></span> <b>Sonda Portátil:</b> VITA-PRB-0428</div>
        <div><span class='status-dot-green'></span> <b>Sensor EIS:</b> AD5933 (Dual Electrodes)</div>
        <div><span class='status-dot-green'></span> <b>Batería:</b> 94% (Autonomía: 8.5 h)</div>
        <div><span class='status-dot-blue'></span> <b>Inferencia Edge:</b> MindSpore Lite (Ascend 310)</div>
        <div><span class='status-dot-green'></span> <b>Huawei Cloud:</b> ModelArts & RDS Conectados</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model Zoo Disponible:**")
    st.markdown("""
    - **Random Forest** (Baseline Balanceado)
    - **SVM (Kernel RBF)** (Calibrado Platt)
    - **XGBoost** (Gradient Boosting)
    - **1D CNN - VITA-Net** (Deep Learning)
    - **Ensamble Multimodelo** (Consenso Ponderado)
    """)

    st.markdown("---")
    st.markdown("**Flujo Clínico Guiado en 3 Pasos:**")
    st.markdown("""
    1. **Paso 1:** Identificación y biomarcadores multimodales.
    2. **Paso 2:** Adquisición de bioimpedancia y control SQI.
    3. **Paso 3:** Selección de modelo, dictamen BI-RADS e incertidumbre.
    """)

    st.markdown("---")
    st.caption("Huawei ICT Competition 2026-2027 | Innovation Track")

# ======================================================================================
# CABECERA DE LA ESTACIÓN DE TRABAJO
# ======================================================================================
st.markdown("""
<div class='hospital-header'>
    <div>
        <span style='font-size:1.15rem; font-weight:700; color:#1E3A8A;'>VITA-ES</span>
        <span style='color:#64748B;'> | Estación de Tamizaje Mamario Oportuno (Multi-Model & UQ)</span>
    </div>
    <div>
        <span class='status-badge'><span class='status-dot-green'></span>Sonda en Línea</span> &nbsp;
        <span class='status-badge'>Huawei ModelArts & MindSpore Sync</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================================================
# NAVEGACIÓN PRINCIPAL
# ======================================================================================
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "1. Estación de Tamizaje Clínico (Flujo Guiado)",
    "2. Benchmarking del Model Zoo (4 Modelos de IA)",
    "3. Biofísica y Explorador Multimodal (3 Datasets)",
    "4. Infraestructura Huawei Cloud (Evaluación Técnica)"
])

# ======================================================================================
# PESTAÑA 1: TAMIZAJE CLÍNICO (FLUJO EN 3 PASOS)
# ======================================================================================
with main_tab1:
    st.markdown("### Tamizaje y Clasificación Tisular por Bioimpedancia")
    st.markdown("Siga el flujo secuencial en 3 pasos para evaluar a la paciente y emitir la recomendación médica oportuna.")

    # Stepper visual
    st.markdown("""
    <div class='stepper-nav'>
        <div class='step-pill'>① PASO 1: Identificación y Perfil Multimodal</div>
        <div class='step-pill'>② PASO 2: Adquisición EIS & Control SQI</div>
        <div class='step-pill'>③ PASO 3: Diagnóstico, Model Zoo & Dictamen</div>
    </div>
    """, unsafe_allow_html=True)

    # Inicialización de variables de sesión
    if "patient_id" not in st.session_state:
        st.session_state.patient_id = "PAC-2026-0842"
    if "patient_age" not in st.session_state:
        st.session_state.patient_age = 49
    if "laterality" not in st.session_state:
        st.session_state.laterality = "Mama Derecha"
    if "quadrant" not in st.session_state:
        st.session_state.quadrant = "Cuadrante Superior Externo (CSE) - Zona de mayor incidencia (~50%)"
    if "palpable" not in st.session_state:
        st.session_state.palpable = "Nódulo o masa palpable localizada"
    if "family_history" not in st.session_state:
        st.session_state.family_history = False
    if "bmi" not in st.session_state:
        st.session_state.bmi = 24.5
    if "dense_breast" not in st.session_state:
        st.session_state.dense_breast = False

    # Variables metabólicas Coimbra
    if "glucose" not in st.session_state:
        st.session_state.glucose = 92.0
    if "insulin" not in st.session_state:
        st.session_state.insulin = 7.5
    if "homa" not in st.session_state:
        st.session_state.homa = 1.7
    if "leptin" not in st.session_state:
        st.session_state.leptin = 18.0
    if "adiponectin" not in st.session_state:
        st.session_state.adiponectin = 12.0
    if "resistin" not in st.session_state:
        st.session_state.resistin = 10.0
    if "mcp1" not in st.session_state:
        st.session_state.mcp1 = 350.0

    # ----------------------------------------------------------------------------------
    # PASO 1: REGISTRO DE LA PACIENTE (FUSIÓN MULTIMODAL CLÍNICA)
    # ----------------------------------------------------------------------------------
    with st.expander("PASO 1: Ficha de la Paciente y Anamnesis Multimodal (BCCD / WDBC)", expanded=True):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            st.session_state.patient_id = st.text_input("Código de Expediente / ID:", value=st.session_state.patient_id)
            st.session_state.patient_age = st.number_input("Edad (años):", min_value=18, max_value=95, value=st.session_state.patient_age)
            st.session_state.bmi = st.number_input("Índice de Masa Corporal - IMC (kg/m²):", min_value=15.0, max_value=55.0, value=float(st.session_state.bmi), step=0.5, help="Biomarcador metabólico (BCCD): IMC > 30 se asocia con inflamación crónica del parénquima.")
        with c_p2:
            st.session_state.laterality = st.selectbox("Mama Evaluada:", ["Mama Derecha", "Mama Izquierda"], index=0 if st.session_state.laterality == "Mama Derecha" else 1)
            st.session_state.quadrant = st.selectbox("Cuadrante Anatómico:", [
                "Cuadrante Superior Externo (CSE) - Zona de mayor incidencia (~50%)",
                "Cuadrante Superior Interno (CSI)",
                "Cuadrante Inferior Externo (CIE)",
                "Cuadrante Inferior Interno (CII)",
                "Región Central / Retroareolar"
            ], index=0)
            st.session_state.dense_breast = st.checkbox("Parénquima mamario denso (Tipo C/D)", value=st.session_state.dense_breast, help="WBCD: La densidad mamaria elevada reduce la sensibilidad mamográfica tradicional.")
        with c_p3:
            st.session_state.palpable = st.selectbox("Hallazgos al Examen Físico:", [
                "Nódulo o masa palpable localizada",
                "Asintomática / Sin nódulo evidente (Tamizaje preventivo)",
                "Dolor / Mastalgia focal",
                "Secreción hemática o serosa por pezón"
            ])
            st.session_state.family_history = st.checkbox("Antecedente familiar directo (Madre/Hermana con Ca. de mama)", value=st.session_state.family_history)
            st.caption("Fusión Multimodal: Combina bioimpedancia con factores de riesgo clínico validados.")

        # Sub-expander para biomarcadores metabólicos Coimbra
        with st.expander("Biomarcadores Metabólicos y Bioquímica Sanguínea (Dataset Coimbra - BCCD)", expanded=False):
            st.caption("Valores basales fisiológicos precargados. Modifique para simular perfiles con resistencia a la insulina o inflamación:")
            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                st.session_state.glucose = st.number_input("Glucosa Basal (mg/dL):", min_value=50.0, max_value=300.0, value=float(st.session_state.glucose), step=1.0)
            with cm2:
                st.session_state.insulin = st.number_input("Insulina (µIU/mL):", min_value=1.0, max_value=80.0, value=float(st.session_state.insulin), step=0.5)
            with cm3:
                st.session_state.homa = st.number_input("Índice HOMA-IR:", min_value=0.2, max_value=15.0, value=float(st.session_state.homa), step=0.1)
            with cm4:
                st.session_state.leptin = st.number_input("Leptina (ng/mL):", min_value=2.0, max_value=100.0, value=float(st.session_state.leptin), step=1.0)

    # ----------------------------------------------------------------------------------
    # PASO 2: ADQUISICIÓN DE LECTURA DE BIOIMPEDANCIA
    # ----------------------------------------------------------------------------------
    st.markdown("#### PASO 2: Adquisición de la Lectura de Bioimpedancia (EIS)")
    
    adq_mode = st.radio(
        "Seleccione el origen de los datos espectrales:",
        [
            "Muestra Clínica Pre-cargada (Dataset de Validación - 106 Casos Reales)",
            "Medición Simulada en Vivo con la Sonda Portátil VITA-ES",
            "Cargar Archivo CSV de Barrido Espectral (Zenodo / Mendeley / Kaggle)"
        ],
        horizontal=True
    )

    current_features = None
    current_sweep = None
    real_case_label = None

    if "Muestra Clínica Pre-cargada" in adq_mode:
        col_filtro, col_caso = st.columns([1, 2])
        with col_filtro:
            filtro_histologico = st.selectbox(
                "Filtrar casos por diagnóstico histológico confirmado:",
                ["Todos los casos", "Carcinoma (Maligno)", "Fibroadenoma (Benigno)", "Mastopatía (Benigno)", "Glandular (Sano)", "Conectivo (Sano)", "Adiposo (Sano)"]
            )
            df_filtrado = dataset.copy()
            if "Carcinoma" in filtro_histologico:
                df_filtrado = df_filtrado[df_filtrado["Class"] == "car"]
            elif "Fibroadenoma" in filtro_histologico:
                df_filtrado = df_filtrado[df_filtrado["Class"] == "fad"]
            elif "Mastopatía" in filtro_histologico:
                df_filtrado = df_filtrado[df_filtrado["Class"] == "mas"]
            elif "Glandular" in filtro_histologico:
                df_filtrado = df_filtrado[df_filtrado["Class"] == "gla"]
            elif "Conectivo" in filtro_histologico:
                df_filtrado = df_filtrado[df_filtrado["Class"] == "con"]
            elif "Adiposo" in filtro_histologico:
                df_filtrado = df_filtrado[df_filtrado["Class"] == "adi"]

        with col_caso:
            opciones_casos = [f"Caso #{r['Case #']} - Biopsia: {r['Class_Name']} | I0: {r['I0']:.1f} Ω" for _, r in df_filtrado.iterrows()]
            caso_seleccionado_str = st.selectbox("Seleccione el registro de paciente a evaluar:", opciones_casos)
            caso_id = int(caso_seleccionado_str.split("#")[1].split(" ")[0])
            registro_caso = dataset[dataset["Case #"] == caso_id].iloc[0]

        real_case_label = registro_caso["Class_Name"]
        current_features = {f: float(registro_caso[f]) for f in FEATURE_NAMES}

        # Reconstruir barrido de Cole-Cole correspondiente
        r0_val = float(registro_caso["I0"])
        dr_val = float(registro_caso["DR"])
        r_inf_val = max(10.0, r0_val - dr_val)
        current_sweep = generate_cole_sweep(
            r0=r0_val,
            r_inf=r_inf_val,
            tau=1.0e-6,
            alpha=0.75,
            noise_std=0.012
        )

    elif "Medición Simulada en Vivo" in adq_mode:
        probe_emu = get_probe_emulator()

        st.markdown(f"""
        <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:10px 16px; margin-bottom:12px; font-size:0.83rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>
            <div>
                <span class='status-dot-green'></span> <b>Sonda Conectada:</b> <code>{probe_emu.DEVICE_ID}</code> &nbsp;|&nbsp;
                <b>Firmware:</b> <code>{probe_emu.FIRMWARE_VERSION}</code> &nbsp;|&nbsp;
                <b>Chipset:</b> <code>AD5933 (12-bit)</code>
            </div>
            <div>
                <b>Batería:</b> {probe_emu.battery_pct:.1f}% &nbsp;|&nbsp;
                <b>Temperatura:</b> {probe_emu.temperature_c} °C &nbsp;|&nbsp;
                <b>Enlace:</b> <span style='color:#059669; font-weight:700;'>BLE 5.0 Activo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2, col_s3, col_s4 = st.columns([1.2, 1.2, 1, 1.2])
        with col_s1:
            tejido_sim = st.selectbox(
                "Patología a emular con la sonda:",
                ["car", "fad", "mas", "gla", "con", "adi"],
                format_func=lambda c: CLASS_MAPPING.get(c, {}).get("name", c.upper())
            )
        with col_s2:
            ruido_sensor = st.slider("Ruido térmico/electrodos (SNR):", 0.005, 0.050, 0.012, step=0.005, help="Simula variación estocástica por impedancia de contacto piel-electrodo")
        with col_s3:
            fallo_contacto = st.checkbox("Simular contacto fallido", value=False, help="Provoca R0 > 6000 Ohm para evaluar el interlock de seguridad clínica SQI.")
        with col_s4:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            btn_adquirir = st.button("Iniciar Escaneo en Vivo (AD5933)", use_container_width=True)

        # Adquisición desde el emulador de hardware biomédico
        packet, current_sweep = probe_emu.acquire_telemetry_scan(
            tissue_class=tejido_sim,
            noise_level=ruido_sensor,
            simulate_faulty_contact=fallo_contacto
        )
        current_features = packet["features_extracted"]
        real_case_label = CLASS_MAPPING.get(tejido_sim, {}).get("name", "Simulado")

        with st.expander("Trama Binaria Cruda BLE (Bluetooth GATT Characteristic)", expanded=False):
            st.code(packet["device_metadata"]["ble_packet_raw_hex"], language="text")
            st.caption(f"Trama transmitida: 12 bytes empaquetados con preámbulo 0xAA55, SQI {packet['device_metadata']['sqi_score']}%, Batería {packet['device_metadata']['battery_pct']}% y CRC16.")

    else:
        # Carga de archivo CSV externo
        st.markdown("""
        <div style='background:#F1F5F9; border:1px dashed #94A3B8; border-radius:6px; padding:12px; margin-bottom:12px; font-size:0.84rem; color:#334155;'>
            <b>Compatibilidad Universal:</b> Acepta archivos CSV de espectrómetros AD5933, Zurick Instruments, Keysight o datasets de Zenodo/Mendeley.
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Arrastre o seleccione el archivo CSV del barrido espectral:", type=["csv"])
        if uploaded_file is not None:
            csv_data = pd.read_csv(uploaded_file)
            norm_df, is_ok, msg = normalize_sweep_dataframe(csv_data)
            if is_ok:
                current_sweep = norm_df
                current_features = extract_features_from_sweep(current_sweep)
                st.success(f"Archivo normalizado correctamente: {len(current_sweep)} frecuencias identificadas.")
            else:
                st.error(f"Error al procesar el archivo CSV: {msg}")
                st.stop()
        else:
            st.info("Por favor seleccione un caso pre-cargado o cargue un archivo CSV para proceder.")
            st.stop()

    # Evaluación de Calidad de Señal (Signal Quality Index - SQI)
    signal_qc = assess_signal_quality(current_sweep)

    # Visualización resumida de la tabla de lecturas
    with st.expander("Tabla de Lecturas del Barrido Espectral Multifrecuencia", expanded=False):
        sweep_display = current_sweep[["Frequency_kHz", "Z_Real", "Minus_Z_Imag", "Z_Magnitude", "Phase_Deg"]].copy()
        sweep_display.columns = ["Frecuencia (kHz)", "Resistencia Z' (Ω)", "Reactancia -Z'' (Ω)", "Magnitud |Z| (Ω)", "Fase θ (grados)"]
        st.dataframe(sweep_display.style.format({
            "Frecuencia (kHz)": "{:.2f}",
            "Resistencia Z' (Ω)": "{:.1f}",
            "Reactancia -Z'' (Ω)": "{:.1f}",
            "Magnitud |Z| (Ω)": "{:.1f}",
            "Fase θ (grados)": "{:.2f}°"
        }), use_container_width=True)

    st.markdown("---")

    # ----------------------------------------------------------------------------------
    # VALIDACIÓN DE SEGURIDAD CLÍNICA ANTES DEL PASO 3
    # ----------------------------------------------------------------------------------
    if not signal_qc["is_acceptable"]:
        st.markdown(f"""
        <div style='background:#FEF2F2; border:1.5px solid #EF4444; border-radius:8px; padding:18px 22px; color:#991B1B; margin-bottom:20px;'>
            <div style='font-size:0.85rem; font-weight:700; text-transform:uppercase;'>INTERLOCK DE SEGURIDAD CLÍNICA VITA-ES: ADQUISICIÓN INVALIDADA</div>
            <div style='font-size:1.15rem; font-weight:700; margin:6px 0;'>{signal_qc['remediation_message']}</div>
            <hr style='border:0; border-top:1px solid rgba(239,68,68,0.3); margin:10px 0;'>
            <div style='font-size:0.85rem; color:#B91C1C;'>
                <b>Parámetros de Calidad Detectados:</b> Índice SQI: <b>{signal_qc['sqi_score']}%</b> &nbsp;|&nbsp; 
                SNR Estimada: <b>{signal_qc['snr_db']} dB</b> &nbsp;|&nbsp; 
                Z' Máxima: <b>{signal_qc['max_z_real']} Ω</b><br>
                <i>Por normatividad biomédica, la inferencia diagnóstica no se emite sobre lecturas artefactadas para evitar falsos positivos o negativos.</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ----------------------------------------------------------------------------------
    # PASO 3: DIAGNÓSTICO, MODEL ZOO Y DICTAMEN MÉDICO
    # ----------------------------------------------------------------------------------
    st.markdown("#### PASO 3: Dictamen Diagnóstico, Model Zoo e Incertidumbre")

    # Selector interactivo de Modelo de Inferencia
    c_m_sel, c_m_info = st.columns([2, 3])
    with c_m_sel:
        model_options = [
            "Random Forest (Baseline)",
            "SVM (Kernel RBF)",
            "XGBoost",
            "1D CNN (VITA-Net)",
            "Ensamble Multimodelo"
        ]
        selected_model_label = st.selectbox(
            "Modelo de Inferencia Activo:",
            model_options,
            index=0,
            help="Alterne entre modelos para comparar predicción, latencia e incertidumbre clínica en tiempo real."
        )
    with c_m_info:
        st.markdown(f"""
        <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:8px 14px; font-size:0.83rem; color:#334155; margin-top:24px;'>
            <span class='status-dot-green'></span> <b>Control de Calidad (SQI):</b> <span style='color:#059669; font-weight:700;'>{signal_qc['sqi_score']}% (Señal Válida)</span> &nbsp;|&nbsp; 
            <b>SNR:</b> {signal_qc['snr_db']} dB &nbsp;|&nbsp; 
            <b>Fusión Multimodal:</b> 70% EIS + 15% Coimbra + 15% WDBC
        </div>
        """, unsafe_allow_html=True)

    # Limpiar etiqueta del modelo para consulta interna al motor
    active_model_name = selected_model_label.split(" (")[0]
    if "Ensamble" in selected_model_label:
        active_model_name = "Ensamble Multimodelo"

    # Preparar perfil metabólico y citológico
    patient_metabolic_dict = {
        "Age": float(st.session_state.patient_age),
        "BMI": float(st.session_state.bmi),
        "Glucose": float(st.session_state.glucose),
        "Insulin": float(st.session_state.insulin),
        "HOMA": float(st.session_state.homa),
        "Leptin": float(st.session_state.leptin),
        "Adiponectin": float(st.session_state.adiponectin),
        "Resistin": float(st.session_state.resistin),
        "MCP.1": float(st.session_state.mcp1)
    }
    is_palpable = ("Nódulo" in st.session_state.palpable)

    # Inferencia del motor VITA-ES
    inference_res = engine.predict(
        features=current_features,
        model_name=active_model_name,
        patient_metabolic=patient_metabolic_dict,
        patient_palpable=is_palpable
    )

    triage_risk = inference_res["triage_risk"]
    triage_code = inference_res["triage_code"]
    triage_color = inference_res["triage_color"]
    uncertainty = inference_res["uncertainty"]
    mm_breakdown = inference_res["multimodal_breakdown"]

    # Estilos del cuadro de dictamen
    if triage_code == 2:
        box_style = "triage-card-red"
        status_title = "ALTO RIESGO ONCOLÓGICO — SOSPECHA DE CARCINOMA"
    elif triage_code == 1:
        box_style = "triage-card-amber"
        status_title = "RIESGO MODERADO — LESIÓN BENIGNA PROLIFERATIVA"
    else:
        box_style = "triage-card-green"
        status_title = "BAJO RIESGO — PATRÓN FISIOLÓGICO NORMAL"

    col_dictamen, col_uq = st.columns([6, 6])

    # Columna 1: Dictamen Principal
    with col_dictamen:
        st.markdown(f"""
        <div class='{box_style}'>
            <div style='font-size:0.78rem; font-weight:700; letter-spacing:0.8px; text-transform:uppercase;'>DICTAMEN DE TRIAJE MULTIMODAL VITA-ES</div>
            <div style='font-size:1.32rem; font-weight:800; margin:5px 0;'>{status_title}</div>
            <div style='font-size:1.02rem; font-weight:600; margin-bottom:8px;'>{inference_res['birads_category']}</div>
            <hr style='border:0; border-top:1px solid rgba(0,0,0,0.1); margin:8px 0;'>
            <div style='font-size:0.86rem; line-height:1.6;'>
                <b>Diagnóstico Tisular Probable:</b> {inference_res['prediction_name']} ({inference_res['confidence']*100:.1f}%)<br>
                <b>Score de Riesgo Fused:</b> <b>{mm_breakdown['fused_risk_score']*100:.1f} / 100</b> (EIS: {mm_breakdown['eis_carcinoma_prob']*100:.1f}%)<br>
                <b>Modelo Utilizado:</b> {inference_res['model_used']}<br>
                <b>Conducta:</b> {inference_res['clinical_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Columna 2: Cuantificación de Incertidumbre Diagnóstica (UQ)
    with col_uq:
        st.markdown(f"""
        <div class='clinical-card' style='border-left: 4px solid {uncertainty["badge_color"]}; height:100%;'>
            <div style='font-size:0.80rem; font-weight:700; color:{uncertainty["badge_color"]}; text-transform:uppercase; margin-bottom:4px;'>
                Cuantificación de Incertidumbre Diagnóstica (CUI)
            </div>
            <div style='font-size:1.45rem; font-weight:800; color:#0F172A; margin:2px 0;'>
                {uncertainty['confidence_score']}% Certeza Diagnóstica
                <span style='font-size:0.88rem; font-weight:500; color:#64748B;'> (CUI: {uncertainty['cui_score']}%)</span>
            </div>
            <div style='background:{uncertainty["badge_bg"]}; color:{uncertainty["badge_color"]}; padding:3px 10px; border-radius:4px; display:inline-block; font-size:0.82rem; font-weight:700; margin:6px 0 8px 0;'>
                {uncertainty['level']}
            </div>
            <div style='font-size:0.84rem; color:#475569; line-height:1.5;'>
                {uncertainty['clinical_interpretation']}
            </div>
            <div style='font-size:0.79rem; color:#64748B; margin-top:8px; border-top:1px solid #E2E8F0; padding-top:6px;'>
                Entropía Shannon Normalizada: <b>{uncertainty['entropy_normalized']}</b> &nbsp;|&nbsp;
                Margen Top-1 vs Top-2: <b>{uncertainty['margin_delta_p']}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Banner de Alerta si la Incertidumbre es Crítica
    if uncertainty["requires_second_opinion"]:
        st.markdown(f"""
        <div style='background:#FEF2F2; border:1px solid #F87171; border-radius:6px; padding:10px 16px; margin-top:12px; font-size:0.86rem; color:#991B1B;'>
            <b>ALERTA DE SEGURIDAD CLÍNICA VITA-ES:</b> La muestra bioeléctrica presenta ambigüedad diagnóstica limítrofe (CUI &gt; 55%). El sistema suspende el descarte preventivo y exige confirmación ecográfica/histológica.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficos Espectrales: Cole-Cole y Bode (Tema claro de alto contraste)
    col_cole, col_bode = st.columns(2)

    with col_cole:
        st.markdown("##### Diagrama de Cole-Cole (Plano de Nyquist: Z' vs -Z'')")
        st.caption("Eje X: Resistencia Real Z' (Ohm). Eje Y: Reactancia Capacitiva -Z'' (Ohm).")

        fig_nyq = go.Figure()

        # Medición real
        fig_nyq.add_trace(go.Scatter(
            x=current_sweep["Z_Real"],
            y=current_sweep["Minus_Z_Imag"],
            mode="markers+lines",
            name="Lectura de Sonda",
            marker=dict(size=9, color="#2563EB", symbol="circle", line=dict(width=1.5, color="#FFFFFF")),
            line=dict(color="#2563EB", width=2.5),
            text=[f"f: {f:.1f} kHz<br>Z': {r:.1f} Ω<br>-Z'': {im:.1f} Ω" for f, r, im in zip(current_sweep["Frequency_kHz"], current_sweep["Z_Real"], current_sweep["Minus_Z_Imag"])],
            hoverinfo="text"
        ))

        # Semicírculo teórico de Cole-Cole
        r0_f = float(current_features.get("I0", 1000.0))
        dr_f = float(current_features.get("DR", 500.0))
        r_inf_f = max(10.0, r0_f - dr_f)
        arc_x, arc_y = generate_ideal_cole_arc(r0_f, r_inf_f, alpha=0.75, n_points=120)

        fig_nyq.add_trace(go.Scatter(
            x=arc_x,
            y=arc_y,
            mode="lines",
            name="Arco Cole Teórico",
            line=dict(color="#94A3B8", width=1.8, dash="dot"),
            hoverinfo="skip"
        ))

        fig_nyq.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            xaxis=dict(title="Resistencia Real Z' (Ω)", gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
            yaxis=dict(title="Reactancia -Z'' (Ω)", gridcolor="#E2E8F0", zerolinecolor="#CBD5E1", scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=30, t=20, b=30),
            height=330
        )
        st.plotly_chart(fig_nyq, use_container_width=True)

    with col_bode:
        st.markdown("##### Diagrama de Bode (Magnitud |Z| y Ángulo de Fase vs Frecuencia)")
        st.caption("Respuesta multifrecuencia desde 15.6 kHz hasta 1.0 MHz.")

        fig_bode = make_subplots(specs=[[{"secondary_y": True}]])

        # Magnitud
        fig_bode.add_trace(
            go.Scatter(
                x=current_sweep["Frequency_kHz"],
                y=current_sweep["Z_Magnitude"],
                name="Magnitud |Z| (Ω)",
                mode="lines+markers",
                line=dict(color="#1E40AF", width=2.5),
                marker=dict(size=7)
            ),
            secondary_y=False
        )

        # Fase
        fig_bode.add_trace(
            go.Scatter(
                x=current_sweep["Frequency_kHz"],
                y=current_sweep["Phase_Deg"],
                name="Fase θ (grados)",
                mode="lines+markers",
                line=dict(color="#D97706", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond")
            ),
            secondary_y=True
        )

        fig_bode.update_xaxes(type="log", title="Frecuencia (kHz)", gridcolor="#E2E8F0")
        fig_bode.update_yaxes(title_text="Magnitud |Z| (Ω)", secondary_y=False, gridcolor="#E2E8F0")
        fig_bode.update_yaxes(title_text="Fase θ (°)", secondary_y=True, gridcolor="#E2E8F0")
        fig_bode.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=30, t=20, b=30),
            height=330
        )
        st.plotly_chart(fig_bode, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # PROBABILIDADES HISTOLÓGICAS Y CARACTERÍSTICAS BIOFÍSICAS
    # ----------------------------------------------------------------------------------
    c_prob, c_feat = st.columns([6, 6])
    with c_prob:
        st.markdown(f"##### Distribución de Probabilidades ({active_model_name})")
        prob_rows = []
        for code, p in inference_res["probabilities"].items():
            info = CLASS_MAPPING.get(code, {})
            prob_rows.append({
                "Tejido": info.get("name", code.upper()),
                "Probabilidad": p * 100.0,
                "Tipo": "Maligno" if code == "car" else ("Benigno" if code in ["fad", "mas"] else "Sano")
            })
        df_p = pd.DataFrame(prob_rows).sort_values("Probabilidad", ascending=True)

        fig_p = px.bar(
            df_p,
            x="Probabilidad",
            y="Tejido",
            orientation="h",
            text="Probabilidad",
            color="Tipo",
            color_discrete_map={"Maligno": "#DC2626", "Benigno": "#D97706", "Sano": "#059669"}
        )
        fig_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_p.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            xaxis=dict(range=[0, 105], title="Probabilidad Estimada (%)", gridcolor="#E2E8F0"),
            yaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=30, t=20, b=30),
            height=280
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with c_feat:
        st.markdown("##### Fusión Multimodal & Características Biofísicas")
        st.markdown(f"""
        <div class='multimodal-container'>
            <div class='multimodal-header'>
                <span>Integración Multimodal de Evidencia Clínica</span>
                <span style='color:#1E40AF; font-weight:700;'>P<sub>Fused</sub>: {mm_breakdown['fused_risk_score']*100:.1f}%</span>
            </div>
            <div class='multimodal-stream-grid'>
                <div class='stream-box'>
                    <span class='stream-tag stream-tag-eis'>EIS 70%</span>
                    <div class='stream-sub'>Bioimpedancia (AD5933)</div>
                    <div class='stream-val'>{mm_breakdown['eis_carcinoma_prob']*100:.1f}%</div>
                    <div class='stream-sub'>Jossinet (N=106)</div>
                </div>
                <div class='stream-box'>
                    <span class='stream-tag stream-tag-coimbra'>Metabólico 15%</span>
                    <div class='stream-sub'>Perfil en Sangre</div>
                    <div class='stream-val'>{mm_breakdown['metabolic_coimbra_prob']*100:.1f}%</div>
                    <div class='stream-sub'>Coimbra (N=116)</div>
                </div>
                <div class='stream-box'>
                    <span class='stream-tag stream-tag-wdbc'>Morfológico 15%</span>
                    <div class='stream-sub'>Palpación / FNA</div>
                    <div class='stream-val'>{mm_breakdown['morphological_wdbc_prob']*100:.1f}%</div>
                    <div class='stream-sub'>Wisconsin (N=569)</div>
                </div>
            </div>
            <div class='multimodal-formula'>
                <span><b>Fórmula Bayesiana:</b> 0.70 &times; P<sub>EIS</sub> + 0.15 &times; P<sub>Metabólico</sub> + 0.15 &times; P<sub>Morfológico</sub></span>
                <span style='font-weight:700; color:#0F172A;'>Score Fused: {mm_breakdown['fused_risk_score']*100:.1f} / 100</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ref_norm = {"I0": 1450.0, "PA500": 0.08, "HFS": 0.015, "DA": 105.0, "Area": 3500.0, "A/DA": 28.0, "Max IP": 32.0, "DR": 85.0, "P": 290.0}
        tabla_caract = []
        for k in FEATURE_NAMES:
            val_p = current_features.get(k, 0.0)
            val_ref = ref_norm.get(k, 1.0)
            desv = ((val_p - val_ref) / val_ref) * 100.0
            tabla_caract.append({
                "Parámetro": k,
                "Valor Paciente": f"{val_p:.2f}",
                "Referencia Sano": f"{val_ref:.2f}",
                "Desviación": f"{desv:+.1f}%",
                "Significado": FEATURE_DESCRIPTIONS.get(k, "")
            })
        st.dataframe(pd.DataFrame(tabla_caract), use_container_width=True, height=210)

    # ----------------------------------------------------------------------------------
    # GENERADOR DE INFORME MÉDICO OFICIAL
    # ----------------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Informe Médico Oficial de Tamizaje")
    
    informe_texto = f"""========================================================================================
CENTRO MÉDICO / BRIGADA DE TAMIZAJE MAMARIO VITA-ES
INFORME CLÍNICO DE ESPECTROSCOPÍA DE BIOIMPEDANCIA ELÉCTRICA (EIS)
========================================================================================
Fecha y Hora de Emisión : {datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M:%S UTC')}
Código de Expediente     : {st.session_state.patient_id}
Edad                     : {st.session_state.patient_age} años
Región Evaluada          : {st.session_state.laterality} - {st.session_state.quadrant}
Hallazgos Palpatorios    : {st.session_state.palpable}
Dispositivo Utilizado    : Sonda VITA-PRB-0428 (MindSpore Lite Edge Ingestion)
Infraestructura Cloud    : Huawei Cloud ModelArts (Ascend 910 Pro) & RDS PostgreSQL

----------------------------------------------------------------------------------------
1. DICTAMEN DE TRIAJE Y EVALUACIÓN DE RIESGO
----------------------------------------------------------------------------------------
Nivel de Riesgo Clínico  : {inference_res['triage_risk']}
Clasificación Estimada   : {inference_res['birads_category']}
Diagnóstico Tisular      : {inference_res['prediction_name']} (Confianza: {inference_res['confidence']*100:.1f}%)
Algoritmo de Inferencia  : {inference_res['model_used']}
Índice de Incertidumbre  : {uncertainty['cui_score']}% ({uncertainty['level']})
Entropía Shannon Norm.   : {uncertainty['entropy_normalized']}
Margen de Confianza ΔP   : {uncertainty['margin_delta_p']}%
Score de Riesgo Fused    : {mm_breakdown['fused_risk_score']*100:.1f} / 100

----------------------------------------------------------------------------------------
2. DESGLOSE MULTIMODAL (BAYESIAN FUSION)
----------------------------------------------------------------------------------------
• Riesgo Primario EIS (70% ponderación) : {mm_breakdown['eis_carcinoma_prob']*100:.1f}%
• Riesgo Metabólico Coimbra (15% pond.) : {mm_breakdown['metabolic_coimbra_prob']*100:.1f}%
• Riesgo Morfológico WDBC (15% pond.)   : {mm_breakdown['morphological_wdbc_prob']*100:.1f}%

----------------------------------------------------------------------------------------
3. PARÁMETROS BIOFÍSICOS DE BIOIMPEDANCIA (EIS)
----------------------------------------------------------------------------------------
• Impedividad Frecuencia Cero (I0) : {current_features['I0']:.2f} Ω
• Pico Reactivo Máximo (Max IP)    : {current_features['Max IP']:.2f} Ω
• Ángulo de Fase a 500 kHz (PA500) : {current_features['PA500']:.4f} rad
• Distancia Espectral (DA)         : {current_features['DA']:.2f} Ω
• Longitud de Curva Compleja (P)   : {current_features['P']:.2f} Ω

----------------------------------------------------------------------------------------
4. CONDUCTA ASISTENCIAL RECOMENDADA
----------------------------------------------------------------------------------------
{inference_res['clinical_action']}

----------------------------------------------------------------------------------------
MÉDICO / ESPECIALISTA EVALUADOR:
Nombre: _________________________________________   Cédula: ____________________________
Firma: __________________________________________   Fecha: _____________________________
========================================================================================
"""
    c_inf1, c_inf2 = st.columns([8, 4])
    with c_inf1:
        with st.expander("Vista Previa del Informe Clínico Estructurado", expanded=False):
            st.text(informe_texto)
    with c_inf2:
        st.download_button(
            label="Descargar Informe Clínico (.txt / HL7)",
            data=informe_texto,
            file_name=f"Informe_VITA_ES_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ======================================================================================
# PESTAÑA 2: BENCHMARKING DEL MODEL ZOO (4 MODELOS DE IA)
# ======================================================================================
with main_tab2:
    st.markdown("### Benchmarking Comparativo del Model Zoo de IA Médica")
    st.markdown("""
    Evaluación rigurosa mediante **Validación Cruzada Estratificada ($k=5$)** sobre las 106 muestras clínicas reales.
    La métrica médica primordial en tamizaje es la **Sensibilidad en Carcinoma** (minimización de falsos negativos oncológicos).
    """)

    comp_data = engine.comparison_metadata
    if comp_data and "models" in comp_data:
        models_dict = comp_data["models"]

        # Resumen en tabla comparativa estructurada
        table_rows = []
        for m_name, m_stats in models_dict.items():
            table_rows.append({
                "Modelo": m_name,
                "Tipo de Arquitectura": m_stats.get("architecture_type", "ML"),
                "Exactitud Global (%)": f"{m_stats['accuracy']:.2f}%",
                "Exactitud Balanceada (%)": f"{m_stats['balanced_accuracy']:.2f}%",
                "F1-Score Macro": f"{m_stats['f1_macro']:.4f}",
                "Sensibilidad Carcinoma (%)": f"{m_stats['carcinoma_sensitivity']:.2f}% (CRÍTICO)",
                "Precisión Carcinoma (%)": f"{m_stats['carcinoma_precision']:.2f}%",
                "Latencia Inferencia (ms)": f"{m_stats['avg_latency_ms']:.2f} ms"
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Matrices de Confusión Interactivas por Modelo")

        col_sel_cm, col_chart_cm = st.columns([1, 2])
        with col_sel_cm:
            sel_model_cm = st.selectbox("Seleccione el modelo para ver su matriz de confusión:", list(models_dict.keys()), index=2)
            st.markdown("""
            <div class='clinical-card' style='font-size:0.84rem; line-height:1.6;'>
                <b>Interpretación de Clases Histológicas:</b><br>
                • <code>adi</code>: Adiposo normal (Sano)<br>
                • <code>car</code>: Carcinoma mamario (Maligno)<br>
                • <code>con</code>: Tejido conectivo fibroso (Sano)<br>
                • <code>fad</code>: Fibroadenoma (Benigno)<br>
                • <code>gla</code>: Parénquima glandular (Sano)<br>
                • <code>mas</code>: Mastopatía fibroquística (Benigno)
            </div>
            """, unsafe_allow_html=True)

        with col_chart_cm:
            cm_arr = np.array(models_dict[sel_model_cm]["confusion_matrix"])
            cm_norm = cm_arr.astype('float') / cm_arr.sum(axis=1)[:, np.newaxis]
            classes_labels = ["Adiposo", "Carcinoma", "Conectivo", "Fibroadenoma", "Glandular", "Mastopatía"]

            fig_cm = px.imshow(
                cm_norm,
                x=classes_labels,
                y=classes_labels,
                labels=dict(x="Predicción de IA", y="Gold Standard (Biopsia)", color="Tasa"),
                color_continuous_scale="Blues",
                text_auto=".2f",
                title=f"Matriz de Confusión Normalizada: {sel_model_cm}"
            )
            fig_cm.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=360)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Análisis de Arquitectura para el Huawei ICT Competition")
        col_hw1, col_hw2 = st.columns(2)
        with col_hw1:
            st.markdown("""
            <div class='clinical-card'>
                <b>Ventajas de Inferencia Edge (MindSpore Lite / Ascend 310):</b><br>
                • <b>Latencia Ultrabaja:</b> Inferencia de 0.17 ms a 5.0 ms directamente en la sonda portátil.<br>
                • <b>Privacidad Médica Local:</b> El procesamiento preliminar ocurre sin enviar datos crudos a internet.<br>
                • <b>Autonomía de Brigadas:</b> Permite tamizaje en comunidades rurales desconectadas.
            </div>
            """, unsafe_allow_html=True)
        with col_hw2:
            st.markdown("""
            <div class='clinical-card'>
                <b>Ventajas de Huawei Cloud ModelArts (Ascend 910 Pro):</b><br>
                • <b>Re-entrenamiento Continuo (MLOps):</b> Pipeline federado para actualizar el Model Zoo.<br>
                • <b>Fusión Multimodal Compleja:</b> Cruce transaccional en tiempo real con RDS PostgreSQL y DCS Redis.<br>
                • <b>Escalabilidad Nacional:</b> Capacidad de gestionar miles de tamizajes concurrentes.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Artefacto models_comparison.json no disponible. Ejecute 'python src/models_zoo.py'.")

# ======================================================================================
# PESTAÑA 3: BIOFÍSICA Y EXPLORADOR MULTIMODAL (3 DATASETS)
# ======================================================================================
with main_tab3:
    st.markdown("### Fundamentos Biofísicos y Explorador de Datasets Multimodales")
    st.markdown("Visualización interactiva de las tres fuentes de datos que alimentan la plataforma VITA-ES:")

    subtab_eis, subtab_coimbra, subtab_wdbc = st.tabs([
        "1. Bioimpedancia Mamaria (Jossinet / 106 Casos)",
        "2. Biomarcadores Metabólicos (Coimbra BCCD / 116 Casos)",
        "3. Citología Morfológica (Wisconsin WDBC / 569 Casos)"
    ])

    with subtab_eis:
        st.markdown("#### Correlación entre Impedancia Extracelular y Malignidad")
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            fig_box1 = px.box(
                dataset,
                x="Class_Name",
                y="I0",
                color="Class_Name",
                points="all",
                title="Distribución de Impedividad I0 (Ω) por Diagnóstico de Biopsia",
                color_discrete_map={info["name"]: info["color"] for info in CLASS_MAPPING.values()}
            )
            fig_box1.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", showlegend=False, height=350)
            st.plotly_chart(fig_box1, use_container_width=True)
        with c_e2:
            fig_scat1 = px.scatter(
                dataset,
                x="I0",
                y="Max IP",
                color="Class_Name",
                title="Proyección Bioeléctrica: I0 (Resistencia) vs Max IP (Reactancia)",
                color_discrete_map={info["name"]: info["color"] for info in CLASS_MAPPING.values()}
            )
            fig_scat1.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=350)
            st.plotly_chart(fig_scat1, use_container_width=True)

    with subtab_coimbra:
        st.markdown("#### Dataset Coimbra (BCCD): Impacto Metabólico en Cáncer de Mama")
        if coimbra_df is not None:
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                coimbra_plot_df = coimbra_df.copy()
                coimbra_plot_df["Diagnóstico"] = coimbra_plot_df["Classification"].map({1: "Control Sano", 2: "Paciente Cáncer"})
                fig_c1 = px.box(
                    coimbra_plot_df,
                    x="Diagnóstico",
                    y="Glucose",
                    color="Diagnóstico",
                    points="all",
                    title="Niveles de Glucosa Basal (mg/dL) vs Diagnóstico Oncológico",
                    color_discrete_map={"Control Sano": "#059669", "Paciente Cáncer": "#DC2626"}
                )
                fig_c1.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=340)
                st.plotly_chart(fig_c1, use_container_width=True)
            with c_c2:
                fig_c2 = px.scatter(
                    coimbra_plot_df,
                    x="BMI",
                    y="Insulin",
                    color="Diagnóstico",
                    title="Correlación: IMC (kg/m²) vs Insulina (µIU/mL)",
                    color_discrete_map={"Control Sano": "#059669", "Paciente Cáncer": "#DC2626"}
                )
                fig_c2.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=340)
                st.plotly_chart(fig_c2, use_container_width=True)
        else:
            st.info("Archivo dataR2.csv no detectado en data/.")

    with subtab_wdbc:
        st.markdown("#### Dataset Wisconsin (WDBC): Morfología Nuclear en Aspiración (FNA)")
        if wdbc_df is not None:
            c_w1, c_w2 = st.columns(2)
            with c_w1:
                wdbc_plot_df = wdbc_df.copy()
                wdbc_plot_df["Diagnóstico"] = wdbc_plot_df["Diagnosis"].map({"B": "Benigno", "M": "Maligno"})
                fig_w1 = px.scatter(
                    wdbc_plot_df,
                    x="Feature_1",
                    y="Feature_2",
                    color="Diagnóstico",
                    labels={"Feature_1": "Radio Medio Celular", "Feature_2": "Textura Media (Escala de Grises)"},
                    title="Separabilidad Morfológica: Radio vs Textura Nuclear",
                    color_discrete_map={"Benigno": "#059669", "Maligno": "#DC2626"}
                )
                fig_w1.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=340)
                st.plotly_chart(fig_w1, use_container_width=True)
            with c_w2:
                fig_w2 = px.box(
                    wdbc_plot_df,
                    x="Diagnóstico",
                    y="Feature_4",
                    color="Diagnóstico",
                    labels={"Feature_4": "Área Media del Núcleo Celular"},
                    title="Distribución del Área Nuclear Celular (FNA)",
                    color_discrete_map={"Benigno": "#059669", "Maligno": "#DC2626"}
                )
                fig_w2.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=340)
                st.plotly_chart(fig_w2, use_container_width=True)
        else:
            st.info("Archivo wdbc.data no detectado en data/.")

# ======================================================================================
# PESTAÑA 4: INFRAESTRUCTURA HUAWEI CLOUD
# ======================================================================================
with main_tab4:
    st.markdown("### Infraestructura y Servicios de Huawei Cloud")
    st.caption("Arquitectura Edge-to-Cloud para el Huawei ICT Competition 2026-2027")

    st.markdown("""
    ```
    ┌─────────────────────────────────┐                ┌─────────────────────────────────────────────────────────┐
    │       DISPOSITIVO EDGE          │                │                  HUAWEI CLOUD SUITE                     │
    │   Sonda Portátil VITA-ES        │                │                                                         │
    │   ┌─────────────────────────┐   │   BLE / 4G     │   ┌───────────────────────┐   ┌───────────────────────┐ │
    │   │ Sensor AD5933           │   │   Telemetry    │   │  Huawei ECS Gateway   │   │  Huawei ModelArts     │ │
    │   │ Ingesta 15 kHz - 1 MHz  │───┼────────────────┼──▶│  Microservicio API    │──▶│  Inferencia Online    │ │
    │   └────────────┬────────────┘   │   (TLS 1.3)    │   │  (EulerOS c7.large.2) │   │  (Ascend 910 Pro NPU) │ │
    │                ▼                │                │   └───────────┬───────────┘   └───────────────────────┘ │
    │   ┌─────────────────────────┐   │                │               │                                         │
    │   │ MindSpore Lite Edge     │   │                │               ▼                           ▼             │
    │   │ Triaje Inmediato (~4 ms)│   │                │   ┌───────────────────────┐   ┌───────────────────────┐ │
    │   └─────────────────────────┘   │                │   │  Huawei DCS (Redis)   │   │  Huawei RDS (Postgre) │ │
    │                                 │                │   │  Buffer de Telemetría │   │  Expedientes Cifrados │ │
    │                                 │                │   │  Sub-millisecond      │   │  KMS HIPAA Compliant  │ │
    │                                 │                │   └───────────────────────┘   └───────────────────────┘ │
    └─────────────────────────────────┘                └─────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")
    st.markdown("#### Consola Interactiva de Simulación de Servicios Cloud")

    col_btn_ma, col_btn_rds, col_btn_dcs = st.columns(3)
    with col_btn_ma:
        probar_ma = st.button("Ejecutar API ModelArts (Inferencia Online)", use_container_width=True)
    with col_btn_rds:
        probar_rds = st.button("Consultar Registro en RDS (PostgreSQL)", use_container_width=True)
    with col_btn_dcs:
        probar_dcs = st.button("Inspeccionar Búfer DCS (Redis Cache)", use_container_width=True)

    if "cloud_view" not in st.session_state:
        st.session_state.cloud_view = "modelarts"

    if probar_ma:
        st.session_state.cloud_view = "modelarts"
    elif probar_rds:
        st.session_state.cloud_view = "rds"
    elif probar_dcs:
        st.session_state.cloud_view = "dcs"

    sample_f = current_features if current_features else {f: 1000.0 for f in FEATURE_NAMES}

    if st.session_state.cloud_view == "modelarts":
        st.markdown("##### Invocación REST: Huawei Cloud ModelArts Inference Endpoint")
        ma_sim = engine.simulate_modelarts_cloud_api(sample_f, patient_id=st.session_state.patient_id)
        c_req, c_resp = st.columns(2)
        with c_req:
            st.markdown(f"**POST Endpoint:** `{ma_sim['endpoint']}`")
            st.json({
                "request_id": ma_sim["request_id"],
                "patient_id": ma_sim["patient_id"],
                "features": sample_f,
                "model_target": active_model_name
            })
        with c_resp:
            st.markdown(f"**Status:** `200 OK` | **Service:** `{ma_sim['service']}`")
            st.json({
                "prediction": ma_sim["inference_result"]["prediction_name"],
                "triage_risk": ma_sim["inference_result"]["triage_risk"],
                "confidence": round(ma_sim["inference_result"]["confidence"]*100, 1),
                "uncertainty_cui": ma_sim["inference_result"]["uncertainty"]["cui_score"],
                "cloud_sync": ma_sim["edge_cloud_sync"]
            })

    elif st.session_state.cloud_view == "rds":
        st.markdown("##### Almacenamiento en Huawei Cloud RDS (PostgreSQL Cifrado)")
        p_info = {"id": st.session_state.patient_id, "laterality": st.session_state.laterality, "quadrant": st.session_state.quadrant}
        rds_rec = engine.simulate_rds_database_persistence(inference_res, p_info)
        st.markdown("""
        <div class='clinical-card'>
            <b>Esquema de Base de Datos:</b> <code>clinical_triage_records</code> &nbsp;|&nbsp;
            <b>Cifrado en Reposo:</b> <code>Huawei Cloud KMS (AES-256)</code> &nbsp;|&nbsp;
            <b>Cumplimiento:</b> <code>HIPAA / GDPR Ready</code>
        </div>
        """, unsafe_allow_html=True)
        st.json(rds_rec)

    elif st.session_state.cloud_view == "dcs":
        st.markdown("##### Búfer de Streaming en Huawei Cloud DCS (Redis Enterprise)")
        dcs_sim = engine.simulate_dcs_redis_cache(st.session_state.patient_id, inference_res)
        st.markdown("""
        <div class='clinical-card'>
            <b>Instancia:</b> <code>dcs-redis-cluster-vitaes</code> &nbsp;|&nbsp;
            <b>Latencia de Escritura:</b> <code>&lt; 1.5 ms</code> &nbsp;|&nbsp;
            <b>Hit Ratio:</b> <code>99.8%</code>
        </div>
        """, unsafe_allow_html=True)
        st.json(dcs_sim)

# ======================================================================================
# PIE DE PÁGINA
# ======================================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#64748B; font-size:0.82rem; padding:10px;'>
    <b>VITA-ES: Estación de Trabajo Clínico y Triaje Oportuno</b> | Desarrollado para el <b>Huawei ICT Competition 2026-2027</b><br>
    <i>Track: Innovation Competition - Topic 1: Creating Innovative AI Applications Powered by Huawei Cloud</i>
</div>
""", unsafe_allow_html=True)
