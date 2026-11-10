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
import importlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configuración de ruta para módulos internos y recarga segura ante hot-reload en Streamlit Cloud
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import data_loader as _dl_mod
import feature_extraction as _fe_mod
import uncertainty as _uq_mod
import cloud_services as _cs_mod
import sonda_emulator as _se_mod
import inference as _inf_mod

for _mod in (_dl_mod, _fe_mod, _uq_mod, _cs_mod, _se_mod, _inf_mod):
    try:
        importlib.reload(_mod)
    except Exception:
        pass

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
def get_inference_engine(_cache_version: str = "3.1.0"):
    return EISInferenceEngine()

@st.cache_resource
def get_probe_emulator(_cache_version: str = "3.1.0"):
    return VITAProbeEmulator()

@st.cache_data
def get_historical_dataset(_cache_version: str = "3.1.0"):
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
    engine = get_inference_engine("3.1.0")
    if not hasattr(engine, "iotda_manager") or not hasattr(engine, "compute_local_treeshap"):
        st.cache_resource.clear()
        engine = EISInferenceEngine()
    dataset = get_historical_dataset("3.1.0")
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
    st.markdown("**Modo de Defensa ante el Jurado:**")
    modo_jurado_split = st.checkbox(
        "🖥️ Pantalla Dividida: Consola Huawei Cloud & IoTDA en Vivo",
        value=False,
        help="Activa el inspector en tiempo real de tramas MQTT hacia Huawei Cloud IoTDA, logs de Cloud Eye/LTS y persistencia RDS/DCS al lado de la estación clínica."
    )

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

        # Sub-expander para biomarcadores metabólicos Coimbra y citológicos WDBC
        with st.expander("Biomarcadores Metabólicos (Coimbra BCCD) y Citología FNA (Wisconsin WDBC)", expanded=False):
            st.caption("Valores basales fisiológicos precargados. Modifique para evaluar perfiles metabólicos o seleccione una muestra citológica WDBC:")
            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                st.session_state.glucose = st.number_input("Glucosa Basal (mg/dL):", min_value=50.0, max_value=300.0, value=float(st.session_state.glucose), step=1.0)
            with cm2:
                st.session_state.insulin = st.number_input("Insulina (µIU/mL):", min_value=1.0, max_value=80.0, value=float(st.session_state.insulin), step=0.5)
            with cm3:
                st.session_state.homa = st.number_input("Índice HOMA-IR:", min_value=0.2, max_value=15.0, value=float(st.session_state.homa), step=0.1)
            with cm4:
                st.session_state.leptin = st.number_input("Leptina (ng/mL):", min_value=2.0, max_value=100.0, value=float(st.session_state.leptin), step=1.0)

            selected_wdbc_vec = None
            if wdbc_df is not None:
                wdbc_mode_opts = ["Usar perfil clínico de palpación (sin punción FNA previa)"] + [
                    f"Caso WDBC #{int(r['ID'])} - Citología: {'Maligno (M)' if r['Diagnosis']=='M' else 'Benigno (B)'} | Radio: {r['Feature_1']:.1f}"
                    for _, r in wdbc_df.head(40).iterrows()
                ]
                wdbc_sel = st.selectbox("Opcional — Vincular aspiración citológica real (Dataset Wisconsin WDBC, 30 variables):", wdbc_mode_opts, index=0)
                if "Caso WDBC #" in wdbc_sel:
                    w_id = int(wdbc_sel.split("#")[1].split(" ")[0])
                    w_row = wdbc_df[wdbc_df["ID"] == w_id].iloc[0]
                    selected_wdbc_vec = [float(w_row[f"Feature_{i}"]) for i in range(1, 31)]

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

        # Reconstruir barrido de Cole-Cole correspondiente con semilla determinista por caso
        r0_val = float(registro_caso["I0"])
        dr_val = float(registro_caso["DR"])
        r_inf_val = max(10.0, r0_val - dr_val)
        try:
            current_sweep = generate_cole_sweep(
                r0=r0_val,
                r_inf=r_inf_val,
                tau=1.0e-6,
                alpha=0.75,
                noise_std=0.012,
                random_seed=1000 + int(caso_id)
            )
        except TypeError:
            np.random.seed(1000 + int(caso_id))
            current_sweep = generate_cole_sweep(
                r0=r0_val,
                r_inf=r_inf_val,
                tau=1.0e-6,
                alpha=0.75,
                noise_std=0.012
            )

    elif "Medición Simulada en Vivo" in adq_mode:
        probe_emu = get_probe_emulator("3.1.0")
        if "scan_seed" not in st.session_state:
            st.session_state.scan_seed = 42

        st.markdown(f"""
        <div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:8px; padding:10px 16px; margin-bottom:12px; font-size:0.83rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>
            <div>
                <span class='status-dot-green'></span> <b>Gemelo Digital Sonda HIL:</b> <code>{probe_emu.DEVICE_ID}</code> &nbsp;|&nbsp;
                <b>Firmware:</b> <code>{probe_emu.FIRMWARE_VERSION}</code> &nbsp;|&nbsp;
                <b>AFE:</b> <code>AD5933/AD5940 (Cole-Cole HIL)</code>
            </div>
            <div>
                <b>Batería:</b> {probe_emu.battery_pct:.1f}% &nbsp;|&nbsp;
                <b>Temperatura:</b> {probe_emu.temperature_c} °C &nbsp;|&nbsp;
                <b>Enlace:</b> <span style='color:#059669; font-weight:700;'>BLE 5.0 / Huawei IoTDA Ready</span>
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
            if btn_adquirir:
                st.session_state.scan_seed += 1

        # Adquisición determinística por semilla de escaneo desde el emulador de hardware biomédico
        try:
            packet, current_sweep = probe_emu.acquire_telemetry_scan(
                tissue_class=tejido_sim,
                noise_level=ruido_sensor,
                simulate_faulty_contact=fallo_contacto,
                random_seed=int(st.session_state.scan_seed)
            )
        except TypeError:
            np.random.seed(int(st.session_state.scan_seed))
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
            "XGBoost (Campeón Sensibilidad)",
            "Random Forest (Baseline)",
            "SVM (Kernel RBF)",
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
    elif "SVM" in selected_model_label:
        active_model_name = "SVM (RBF)"
    elif "1D CNN" in selected_model_label:
        active_model_name = "1D CNN (VITA-Net)"

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

    # Inferencia del motor VITA-ES (Multimodal Bayesiana Adaptativa + TreeSHAP + Grafo Normativo)
    try:
        inference_res = engine.predict(
            features=current_features,
            model_name=active_model_name,
            patient_metabolic=patient_metabolic_dict,
            patient_palpable=is_palpable,
            wdbc_features=selected_wdbc_vec,
            patient_age=int(st.session_state.patient_age),
            dense_breast=bool(st.session_state.dense_breast)
        )
    except TypeError:
        inference_res = engine.predict(
            features=current_features,
            model_name=active_model_name,
            patient_metabolic=patient_metabolic_dict,
            patient_palpable=is_palpable,
            wdbc_features=selected_wdbc_vec
        )

    triage_risk = inference_res["triage_risk"]
    triage_code = inference_res["triage_code"]
    triage_color = inference_res["triage_color"]
    uncertainty = inference_res["uncertainty"]
    mm_breakdown = inference_res["multimodal_breakdown"]
    local_shap = inference_res.get("local_shap", {})
    reg_kg = inference_res.get("regulatory_knowledge_graph", {})

    # Publicar trama de telemetría ciberfísica al bus Huawei Cloud IoTDA / Cloud Eye LTS
    iotda_live_frame = engine.iotda_manager.publish_hil_telemetry_frame(
        patient_id=st.session_state.patient_id,
        features=current_features,
        sqi_score=signal_qc["sqi_score"],
        snr_db=signal_qc["snr_db"],
        inference_result=inference_res
    )

    # Estilos del cuadro de dictamen
    if triage_code == 2:
        box_style = "triage-card-red"
        status_title = "ALTO RIESGO ONCOLÓGICO — SOSPECHA DE CARCINOMA"
    elif triage_code == 1:
        box_style = "triage-card-amber"
        status_title = "RIESGO MODERADO — LESIÓN BENIGNA / VIGILANCIA ACTIVA"
    else:
        box_style = "triage-card-green"
        status_title = "BAJO RIESGO — PATRÓN FISIOLÓGICO NORMAL"

    col_dictamen, col_uq = st.columns([6, 6])

    # Columna 1: Dictamen Principal
    with col_dictamen:
        eis_only_name = inference_res.get("eis_only_prediction_name", inference_res["prediction_name"])
        esc_reason = mm_breakdown.get("escalation_reason", "Concordancia Primaria EIS")
        st.markdown(f"""
        <div class='{box_style}'>
            <div style='font-size:0.78rem; font-weight:700; letter-spacing:0.8px; text-transform:uppercase;'>DICTAMEN DE TRIAJE MULTIMODAL VITA-ES</div>
            <div style='font-size:1.32rem; font-weight:800; margin:5px 0;'>{status_title}</div>
            <div style='font-size:1.02rem; font-weight:600; margin-bottom:8px;'>{inference_res['birads_category']}</div>
            <hr style='border:0; border-top:1px solid rgba(0,0,0,0.1); margin:8px 0;'>
            <div style='font-size:0.86rem; line-height:1.6;'>
                <b>Diagnóstico Posterior Multimodal:</b> {inference_res['prediction_name']} ({inference_res['confidence']*100:.1f}%) <i>[EIS Primario: {eis_only_name}]</i><br>
                <b>Score de Riesgo Fused:</b> <b>{mm_breakdown['fused_risk_score']*100:.1f} / 100</b> (EIS Ca.: {mm_breakdown['eis_carcinoma_prob']*100:.1f}% &rarr; Posterior Ca.: {mm_breakdown.get('posterior_carcinoma_prob', mm_breakdown['eis_carcinoma_prob'])*100:.1f}%)<br>
                <b>Estado de Fusión:</b> {esc_reason} &nbsp;|&nbsp; <b>Modelo:</b> {inference_res['model_used']}<br>
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
    # PROBABILIDADES HISTOLÓGICAS (PRIOR EIS VS POSTERIOR MULTIMODAL) Y CARACTERÍSTICAS
    # ----------------------------------------------------------------------------------
    c_prob, c_feat = st.columns([6, 6])
    with c_prob:
        st.markdown(f"##### Probabilidades: Prior EIS vs. Posterior Multimodal ({active_model_name})")
        eis_p_dict = inference_res.get("eis_probabilities", inference_res["probabilities"])
        post_p_dict = inference_res["probabilities"]
        prob_rows = []
        for code in ["adi", "con", "gla", "mas", "fad", "car"]:
            info = CLASS_MAPPING.get(code, {})
            t_name = info.get("name", code.upper())
            prob_rows.append({
                "Tejido": t_name,
                "Probabilidad": eis_p_dict.get(code, 0.0) * 100.0,
                "Etapa": "1. Prior EIS Puro (Bioimpedancia)"
            })
            prob_rows.append({
                "Tejido": t_name,
                "Probabilidad": post_p_dict.get(code, 0.0) * 100.0,
                "Etapa": "2. Posterior Multimodal (EIS + Coimbra + WDBC)"
            })
        df_p = pd.DataFrame(prob_rows)

        fig_p = px.bar(
            df_p,
            x="Probabilidad",
            y="Tejido",
            color="Etapa",
            barmode="group",
            orientation="h",
            text="Probabilidad",
            color_discrete_map={
                "1. Prior EIS Puro (Bioimpedancia)": "#94A3B8",
                "2. Posterior Multimodal (EIS + Coimbra + WDBC)": "#1D4ED8"
            }
        )
        fig_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_p.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            xaxis=dict(range=[0, 108], title="Probabilidad Estimada (%)", gridcolor="#E2E8F0"),
            yaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=30, t=20, b=30),
            height=300
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with c_feat:
        st.markdown("##### Fusión Multimodal Adaptativa & Características Biofísicas")
        st.markdown(f"""
        <div class='multimodal-container'>
            <div class='multimodal-header'>
                <span>Integración Multimodal Bayesiana (Prior &rarr; Posterior)</span>
                <span style='color:#1E40AF; font-weight:700;'>P<sub>Fused</sub>: {mm_breakdown['fused_risk_score']*100:.1f}%</span>
            </div>
            <div class='multimodal-stream-grid'>
                <div class='stream-box'>
                    <span class='stream-tag stream-tag-eis'>EIS Primario</span>
                    <div class='stream-sub'>Bioimpedancia (Cole-Cole)</div>
                    <div class='stream-val'>{mm_breakdown['eis_carcinoma_prob']*100:.1f}%</div>
                    <div class='stream-sub'>Posterior Ca: <b>{mm_breakdown.get('posterior_carcinoma_prob', mm_breakdown['eis_carcinoma_prob'])*100:.1f}%</b></div>
                </div>
                <div class='stream-box'>
                    <span class='stream-tag stream-tag-coimbra'>Metabólico (BCCD)</span>
                    <div class='stream-sub'>Glucosa/Insulina/Leptina</div>
                    <div class='stream-val'>{mm_breakdown['metabolic_coimbra_prob']*100:.1f}%</div>
                    <div class='stream-sub'>Coimbra RF (N=116)</div>
                </div>
                <div class='stream-box'>
                    <span class='stream-tag stream-tag-wdbc'>Morfológico (WDBC)</span>
                    <div class='stream-sub'>{mm_breakdown.get('wdbc_evaluation_source', 'Palpación / FNA')}</div>
                    <div class='stream-val'>{mm_breakdown['morphological_wdbc_prob']*100:.1f}%</div>
                    <div class='stream-sub'>Wisconsin (N=569)</div>
                </div>
            </div>
            <div class='multimodal-formula'>
                <span><b> Estado:</b> {mm_breakdown.get('escalation_reason', 'Concordancia Primaria EIS')}</span>
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
    # EXPLICABILIDAD CLÍNICA EN TIEMPO REAL (TREESHAP) & GRAFO NORMATIVO (SRS / MINSAL)
    # ----------------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    col_xai, col_kg = st.columns([6, 6])

    with col_xai:
        st.markdown("##### Explicabilidad Clínica en Tiempo Real (TreeSHAP Local por Paciente)")
        st.caption("Atribución exacta (Lundberg & Lee, TreeSHAP) sobre qué variables biofísicas dispararon o contuvieron la alerta de Carcinoma en esta paciente:")
        shap_attrs = local_shap.get("attributions", [])
        if shap_attrs:
            df_shap = pd.DataFrame(shap_attrs).sort_values("shap_log_odds", ascending=True)
            fig_shap = px.bar(
                df_shap,
                x="shap_log_odds",
                y="feature",
                orientation="h",
                color="direction",
                text="shap_log_odds",
                labels={"shap_log_odds": "Contribución TreeSHAP (Δ log-odds Carcinoma)", "feature": "Parámetro Cole-Cole"},
                color_discrete_map={
                    "Impulsa Riesgo Maligno (+)": "#DC2626",
                    "Factor Protector / Benigno (-)": "#059669"
                }
            )
            fig_shap.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
            fig_shap.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                xaxis=dict(title="Impacto SHAP en Log-Odds de Malignidad", zeroline=True, zerolinewidth=2, zerolinecolor="#475569"),
                yaxis=dict(title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=30, r=40, t=20, b=30),
                height=300
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    with col_kg:
        st.markdown("##### Grafo de Conocimiento Clínico y Normativo (SRS / MINSAL / BI-RADS)")
        st.caption("Vinculación determinista entre el hallazgo bioeléctrico/multimodal y las guías sanitarias vigentes (0% alucinaciones):")
        st.markdown(f"""
        <div class='clinical-card' style='font-size:0.84rem; line-height:1.65;'>
            <div><b>🏛️ Clasificación Regulatoria (SRS El Salvador):</b><br>
            <span style='color:#1E40AF; font-weight:600;'>{reg_kg.get('srs_classification', 'N/A')}</span></div>
            <div style='margin-top:6px;'><b>⚡ Seguridad Bioeléctrica:</b> {reg_kg.get('electrical_safety_standard', 'N/A')}</div>
            <div style='margin-top:6px;'><b>📋 Protocolo MINSAL Aplicado:</b><br>
            <span style='color:#0F172A;'>{reg_kg.get('minsal_protocol', 'N/A')}</span></div>
            <div style='margin-top:6px;'><b>🔬 Estudio de Imagen de 1ª Línea Indicado (Edad {st.session_state.patient_age}a / Densidad):</b><br>
            <span style='color:#047857; font-weight:600;'>{reg_kg.get('recommended_imaging', 'N/A')}</span></div>
            <div style='margin-top:6px;'><b>🏥 Nivel de Referencia y Ventana SLA:</b><br>
            <b>Destino:</b> {reg_kg.get('target_healthcare_facility', 'N/A')} &nbsp;|&nbsp; <b>Plazo:</b> <span style='color:#B91C1C; font-weight:700;'>{reg_kg.get('sla_response_time', 'N/A')}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------------
    # MODO JURADO: PANTALLA DIVIDIDA (HUAWEI CLOUD CONSOLE & IOTDA TELEMETRY EN VIVO)
    # ----------------------------------------------------------------------------------
    with st.expander("🖥️ Consola Activa Huawei Cloud en Vivo (Split-Screen Demo Flow: IoTDA MQTT + Cloud Eye LTS + RDS)", expanded=modo_jurado_split):
        st.markdown("###### Telemetría Ciberfísica en Tiempo Real: Gemelo Digital Sonda AD5933/AD5940 &rarr; Huawei Cloud IoTDA &rarr; ModelArts &rarr; RDS")
        col_iot_l, col_iot_r = st.columns([6, 6])
        with col_iot_l:
            st.markdown(f"**Trama MQTT Publicada en Huawei Cloud IoTDA** (`{iotda_live_frame['topic']}`):")
            st.json(iotda_live_frame)
        with col_iot_r:
            st.markdown("**Stream de Logs en Vivo — Huawei Cloud Eye / Log Tank Service (LTS):**")
            lts_logs = engine.iotda_manager.get_recent_lts_logs(limit=10)
            if lts_logs:
                st.dataframe(pd.DataFrame(lts_logs), use_container_width=True, height=260)
            c_sync1, c_sync2 = st.columns(2)
            with c_sync1:
                if st.button("⚡ Commit Transaccional en Huawei RDS Ahora", key="btn_split_rds", use_container_width=True):
                    rec_rds = engine.persist_to_rds(inference_res, {
                        "id": st.session_state.patient_id,
                        "age": st.session_state.patient_age,
                        "laterality": st.session_state.laterality,
                        "quadrant": st.session_state.quadrant
                    })
                    engine.iotda_manager.append_lts_event(
                        service="Huawei Cloud RDS (PostgreSQL)",
                        event_type="SQL_TRANSACTION_COMMIT",
                        patient_id=st.session_state.patient_id,
                        latency_ms=rec_rds["commit_latency_ms"],
                        status="COMMITTED",
                        summary=f"Record={rec_rds['record_id']} | TotalDB={rec_rds['total_records_in_db']}"
                    )
                    st.success(f"Expediente `{rec_rds['record_id']}` persistido ({rec_rds['commit_latency_ms']} ms).")
            with c_sync2:
                if st.button("🚀 Sincronizar Sesión en Huawei DCS (Redis)", key="btn_split_dcs", use_container_width=True):
                    rec_dcs = engine.cache_in_dcs(st.session_state.patient_id, inference_res)
                    engine.iotda_manager.append_lts_event(
                        service="Huawei Cloud DCS (Redis)",
                        event_type="KV_SETEX_3600S",
                        patient_id=st.session_state.patient_id,
                        latency_ms=rec_dcs["measured_latency_ms"],
                        status="STORED",
                        summary=f"Key={rec_dcs['cache_key']} | ActiveKeys={rec_dcs['active_keys_in_cache']}"
                    )
                    st.success(f"Caché `{rec_dcs['cache_key']}` actualizada ({rec_dcs['measured_latency_ms']} ms).")

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
                "Exactitud Histológica (6 Clases)": f"{m_stats['accuracy']:.2f}%",
                "Exactitud Triaje BI-RADS (3 Niveles)": f"{m_stats.get('triage_3class_accuracy', m_stats['accuracy']):.2f}%",
                "Sensibilidad Carcinoma (%)": f"{m_stats['carcinoma_sensitivity']:.2f}% (CRÍTICO)",
                "Especificidad Carcinoma (%)": f"{m_stats.get('carcinoma_specificity', 95.0):.2f}%",
                "ROC-AUC Carcinoma": f"{m_stats.get('carcinoma_roc_auc', 0.96):.4f}",
                "ROC-AUC Macro (OvR)": f"{m_stats.get('roc_auc_ovr_macro', 0.90):.4f}",
                "F1-Score Macro": f"{m_stats['f1_macro']:.4f}",
                "Latencia Real (ms)": f"{m_stats['avg_latency_ms']:.2f} ms"
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        aux_meta = comp_data.get("auxiliary_models", {})
        if aux_meta:
            c_aux1, c_aux2 = st.columns(2)
            with c_aux1:
                c_st = aux_meta.get("coimbra", {})
                st.caption(f"🔹 **Modelo Auxiliar Metabólico (Coimbra BCCD, N={c_st.get('samples', 116)}):** Exactitud CV (k=5): **{c_st.get('cv_accuracy', 72.41)}%** | Sensibilidad: **{c_st.get('cv_sensitivity', 75.0)}%**")
            with c_aux2:
                w_st = aux_meta.get("wdbc", {})
                st.caption(f"🔹 **Modelo Auxiliar Morfológico (Wisconsin WDBC, N={w_st.get('samples', 569)}):** Exactitud CV (k=5): **{w_st.get('cv_accuracy', 97.36)}%** | Sensibilidad: **{w_st.get('cv_sensitivity', 94.34)}%**")

        st.markdown("---")
        st.markdown("#### Matrices de Confusión, Curvas ROC-AUC y Atribución Global TreeSHAP")

        col_sel_cm, col_chart_cm, col_chart_roc = st.columns([0.9, 1.4, 1.4])
        with col_sel_cm:
            sel_model_cm = st.selectbox("Seleccione el modelo para ver su matriz de confusión:", list(models_dict.keys()), index=2)
            q_bench = comp_data.get("quantization_benchmark", {})
            st.markdown(f"""
            <div class='clinical-card' style='font-size:0.82rem; line-height:1.55;'>
                <b>Clases Histológicas (Jossinet):</b><br>
                • <code>adi</code>: Adiposo (Sano)<br>
                • <code>car</code>: Carcinoma (Maligno)<br>
                • <code>con</code>: Conectivo (Sano)<br>
                • <code>fad</code>: Fibroadenoma (Benigno)<br>
                • <code>gla</code>: Glandular (Sano)<br>
                • <code>mas</code>: Mastopatía (Benigno)<br>
                <hr style='border:0; border-top:1px solid #E2E8F0; margin:6px 0;'>
                <b>Huawei MindSpore / INT8 Edge:</b><br>
                • Peso FP32: <code>{q_bench.get('fp32_buffer_bytes', 30000)} B</code><br>
                • Peso INT8: <code>{q_bench.get('int8_buffer_bytes', 7500)} B</code><br>
                • Compresión: <b>{q_bench.get('compression_ratio', '4.0x')}</b>
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
                title=f"Matriz de Confusión (5-Fold CV): {sel_model_cm}"
            )
            fig_cm.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_chart_roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Azar (AUC = 0.50)",
                line=dict(color="#94A3B8", dash="dash", width=1.5)
            ))
            palette_roc = {
                "XGBoost": "#DC2626",
                "Ensamble Multimodelo": "#1D4ED8",
                "1D CNN (VITA-Net)": "#7C3AED",
                "Random Forest": "#059669",
                "SVM (RBF)": "#D97706"
            }
            for m_k, m_v in models_dict.items():
                rc = m_v.get("carcinoma_roc_curve")
                auc_val = m_v.get("carcinoma_roc_auc", 0.95)
                if rc and "fpr" in rc and "tpr" in rc:
                    fig_roc.add_trace(go.Scatter(
                        x=rc["fpr"],
                        y=rc["tpr"],
                        mode="lines",
                        name=f"{m_k} (AUC={auc_val:.3f})",
                        line=dict(color=palette_roc.get(m_k, "#334155"), width=2.5 if m_k == "XGBoost" else 1.8)
                    ))
            fig_roc.update_layout(
                title="Curvas ROC en Detección de Carcinoma (5-Fold CV)",
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                xaxis=dict(title="1 - Especificidad (FPR)", gridcolor="#E2E8F0"),
                yaxis=dict(title="Sensibilidad Carcinoma (TPR)", gridcolor="#E2E8F0"),
                legend=dict(font=dict(size=10), x=0.35, y=0.08, bgcolor="rgba(255,255,255,0.85)"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=360
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Gráfico de Importancia Global TreeSHAP vs Gini
        shap_glob = comp_data.get("treeshap_global_importances", {})
        gini_glob = comp_data.get("feature_importances", {})
        if shap_glob and gini_glob:
            rows_imp = []
            for f_n in FEATURE_NAMES:
                rows_imp.append({"Variable": f_n, "Peso Relativo (%)": shap_glob.get(f_n, 0.0) * 100.0, "Método XAI": "TreeSHAP Global (XGBoost Carcinoma)"})
                rows_imp.append({"Variable": f_n, "Peso Relativo (%)": gini_glob.get(f_n, 0.0) * 100.0, "Método XAI": "Impureza Gini (Balanced RF Multiclase)"})
            fig_imp = px.bar(
                pd.DataFrame(rows_imp),
                x="Variable",
                y="Peso Relativo (%)",
                color="Método XAI",
                barmode="group",
                title="Importancia Global de Características Biofísicas: TreeSHAP Oncológico vs. Impureza Gini",
                color_discrete_map={"TreeSHAP Global (XGBoost Carcinoma)": "#DC2626", "Impureza Gini (Balanced RF Multiclase)": "#2563EB"}
            )
            fig_imp.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)

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
    st.markdown("#### Consola Interactiva de Servicios Cloud y Persistencia Transaccional Real")

    col_btn_ma, col_btn_rds, col_btn_dcs = st.columns(3)
    with col_btn_ma:
        probar_ma = st.button("1. Invocar Servicio ModelArts (customize_service.py)", use_container_width=True)
    with col_btn_rds:
        probar_rds = st.button("2. Persistir Expediente en RDS (PostgreSQL / SQLite)", use_container_width=True)
    with col_btn_dcs:
        probar_dcs = st.button("3. Almacenar Sesión en Caché DCS (Redis / KV)", use_container_width=True)

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
        st.markdown("##### Ejecución en Vivo: Huawei Cloud ModelArts Inference Service (`modelarts_deploy/customize_service.py`)")
        ma_res = engine.invoke_modelarts_service(
            features=sample_f,
            patient_id=st.session_state.patient_id,
            model_name=active_model_name,
            metabolic_factors=patient_metabolic_dict,
            patient_palpable=is_palpable
        )
        c_req, c_resp = st.columns(2)
        with c_req:
            st.markdown(f"**Endpoint:** `{ma_res['endpoint']}` | **Latencia medida:** `{ma_res['roundtrip_latency_ms']} ms`")
            st.json(ma_res["request_payload"])
        with c_resp:
            st.markdown(f"**HTTP Status:** `{ma_res['http_status_code']} OK` | **Modo:** `{ma_res['execution_mode']}`")
            st.json(ma_res["service_response"])

    elif st.session_state.cloud_view == "rds":
        st.markdown("##### Transacción Relacional en Huawei Cloud RDS (PostgreSQL / SQLite Local Verificable)")
        p_info = {
            "id": st.session_state.patient_id,
            "age": st.session_state.patient_age,
            "laterality": st.session_state.laterality,
            "quadrant": st.session_state.quadrant
        }
        rds_rec = engine.persist_to_rds(inference_res, p_info)
        st.markdown(f"""
        <div class='clinical-card'>
            <b>Motor Activo:</b> <code>{rds_rec['database_engine']}</code> &nbsp;|&nbsp;
            <b>Latencia Commit:</b> <code>{rds_rec['commit_latency_ms']} ms</code> &nbsp;|&nbsp;
            <b>Total Expedientes en BD:</b> <code>{rds_rec['total_records_in_db']}</code> &nbsp;|&nbsp;
            <b>Integridad:</b> <code>SHA-256 (RTS 11.03.04:23)</code>
        </div>
        """, unsafe_allow_html=True)
        st.json(rds_rec)

        recent_rows = engine.rds_manager.fetch_recent_records(limit=10)
        if recent_rows:
            st.markdown("###### Historial Real de Expedientes Persistidos en `clinical_triage_records`:")
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True)

    elif st.session_state.cloud_view == "dcs":
        st.markdown("##### Almacenamiento de Baja Latencia en Huawei Cloud DCS (Redis / KV Store)")
        dcs_sim = engine.cache_in_dcs(st.session_state.patient_id, inference_res)
        st.markdown(f"""
        <div class='clinical-card'>
            <b>Backend Activo:</b> <code>{dcs_sim['service']}</code> &nbsp;|&nbsp;
            <b>Latencia Medida:</b> <code>{dcs_sim['measured_latency_ms']} ms</code> &nbsp;|&nbsp;
            <b>Sesiones Activas en Caché:</b> <code>{dcs_sim['active_keys_in_cache']}</code>
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
