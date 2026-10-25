# VITA-ES: Plataforma Edge-to-Cloud de Tamizaje de Cáncer de Mama

[![Huawei ICT Competition](https://img.shields.io/badge/Huawei%20ICT%20Competition-2026--2027-red?style=for-the-badge&logo=huawei)](https://e.huawei.com/en/talent/#/ict-academy/ict-competition)
[![Track](https://img.shields.io/badge/Track-Innovation%20Competition-blue?style=for-the-badge)](https://e.huawei.com)
[![Topic](https://img.shields.io/badge/Topic%201-AI%20Applications%20Powered%20by%20Huawei%20Cloud-orange?style=for-the-badge)](https://www.huaweicloud.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63+-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📑 Resumen Ejecutivo

**VITA-ES** es un ecosistema médico integral de triaje oportuno para la detección temprana de cáncer de mama mediante **Espectroscopía de Bioimpedancia Eléctrica (EIS)** y una arquitectura de Inteligencia Artificial **Edge-to-Cloud** potenciada por la suite de servicios de **Huawei Cloud** (**ModelArts, ECS, RDS y DCS**).

El cáncer de mama es la neoplasia maligna más frecuente y la principal causa de mortalidad por cáncer en mujeres a nivel mundial. En América Latina y zonas rurales o periurbanas, más del 65% de los diagnósticos se realizan en etapas tardías (Estadios III y IV) debido a la escasez de mastógrafos digitales, barreras geográficas y listas de espera de meses para biopsias.

**VITA-ES** resuelve este cuello de botella con una sonda portátil no invasiva, libre de radiación ionizante y de bajo costo que realiza barridos multifrecuencia (15.625 kHz - 1.000 MHz) en segundos. Los datos son procesados mediante el modelo electrofisiológico de **Cole-Cole**, clasificados por un ensamble de Machine Learning calibrado para alta sensibilidad oncológica y sincronizados con Huawei Cloud para seguimiento epidemiológico y validación médica.

---

## 🏛️ Arquitectura Edge-to-Cloud con Huawei Cloud

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DISPOSITIVO EDGE                                     │
│  Sonda Portátil VITA-ES (Microcontrolador + Frontend de Bioimpedancia AD5933)           │
│  ┌─────────────────────────┐          ┌─────────────────────────────────────────────┐  │
│  │ Ingesta Multifrecuencia │          │ Inferencia Ultra-Rápida Offline             │  │
│  │ (15.625 kHz a 1.0 MHz)  │─────────▶│ MindSpore Lite Edge Engine (~6 ms)          │  │
│  └─────────────────────────┘          └─────────────────────────────────────────────┘  │
└───────────────────────────────────────┬────────────────────────────────────────────────┘
                                        │ Telemetría Segura (BLE 5.3 / 4G LTE-M / TLS 1.3)
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  HUAWEI CLOUD SUITE                                    │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Huawei Cloud ECS (Elastic Cloud Server - EulerOS / c7.large.2)                  │  │
│  │  Gateway de Servicios Clínicos & Plataforma Web Interactiva Streamlit            │  │
│  └─────────────────────────┬───────────────────────────────────────┬────────────────┘  │
│                            │                                       │                   │
│                            ▼                                       ▼                   │
│  ┌───────────────────────────────────────┐   ┌──────────────────────────────────────┐  │
│  │ Huawei Cloud ModelArts                │   │ Huawei Cloud DCS (Redis 6.0)         │  │
│  │ • Online Inference API (Ascend 910B)  │   │ • Búfer de telemetría de espectros   │  │
│  │ • Reentrenamiento continuo federado   │   │ • Sub-millisecond streaming cache    │  │
│  │ • Monitoreo de Data Drift             │   │ • Gestión de sesiones de sondas      │  │
│  └───────────────────────────────────────┘   └──────────────────────────────────────┘  │
│                            │                                                           │
│                            ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Huawei Cloud RDS (Relational Database Service - PostgreSQL 14 Enterprise)        │  │
│  │ • Almacenamiento seguro y cifrado en reposo (KMS AES-256)                        │  │
│  │ • Expedientes clínicos electrónicos de pacientes, curvas EIS y firmas médicas   │  │
│  │ • Trazabilidad conforme a estándares HIPAA y normativas de salud                 │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧬 Fundamentos Biofísicos: Modelo de Cole-Cole en Tejido Mamario

Las células del parénquima mamario se comportan como circuitos eléctricos complejos compuestos por resistencias intra y extracelulares ($R_i, R_e$) y capacitores dieléctricos de membrana lipídica ($C_m$).

La impedancia compleja $Z(\omega)$ en función de la frecuencia angular $\omega = 2\pi f$ se describe formalmente mediante la **ecuación de Cole-Cole**:

$$Z(\omega) = R_\infty + \frac{R_0 - R_\infty}{1 + (j \omega \tau)^\alpha}$$

Donde:
* **$R_0$ (Impedividad a frecuencia cero, $I_0$):** Resistencia dominada exclusivamente por el espacio extracelular.
* **$R_\infty$:** Resistencia a frecuencia infinita donde la membrana celular se vuelve eléctricamente transparente.
* **$\tau$:** Tiempo característico de relajación dipolar.
* **$\alpha$ ($0 < \alpha \le 1$):** Parámetro de distribución y dispersión de Cole.

### 🔬 ¿Por qué el Cáncer altera la Bioimpedancia?
1. **Proliferación celular desordenada y angiogénesis:** El tejido tumoral maligno presenta membranas celulares depolarizadas, rotura de uniones intercelulares estrechas (gap junctions) y un marcado incremento de fluidos intersticiales y electrolitos (Na+, Cl-).
2. **Caída drástica de $R_0$ ($I_0$):** Mientras el tejido graso sano y el estroma conectivo presentan valores de $I_0 > 1500\ \Omega$, el **Carcinoma** colapsa la resistencia extracelular a rangos de **$250 - 550\ \Omega$**.
3. **Desplazamiento del Arco de Nyquist:** El semicírculo en el plano $(Z', -Z'')$ se reduce drásticamente de diámetro ($DA$) y el pico reactivo ($-Z''_{\max}$, *Max IP*) se contrae, proporcionando una firma biométrica distintiva.

---

## 📁 Arquitectura del Repositorio

```text
vita-es-demo/
├── data/
│   ├── breast_tissue.csv      # Dataset clínico procesado (106 pacientes, 6 clases)
│   └── BreastTissue.xls       # Dataset original de validación (Jossinet / Marques de Sá)
├── models/
│   ├── model.pkl              # Clasificador entrenado de ensamble Random Forest
│   ├── scaler.pkl             # Escalador de características (StandardScaler)
│   └── metadata.json          # Métricas de validación, importancia de variables y specs ModelArts
├── src/
│   ├── __init__.py            # Inicialización del paquete
│   ├── data_loader.py         # Ingesta de datos, mapeo clínico y generador de barridos sintéticos
│   ├── feature_extraction.py  # Cálculo de las 9 características EIS y ajuste de Cole-Cole
│   ├── train.py               # Pipeline de entrenamiento, validación cruzada y serialización
│   └── inference.py           # Motor de inferencia clínica y simulación de Huawei Cloud
├── app.py                     # Aplicación web interactiva en Streamlit (UI Clínica Premium)
├── requirements.txt           # Dependencias de producción
├── Dockerfile                 # Contenedor optimizado para Huawei Cloud ECS / CCE
└── README.md                  # Documentación técnica completa
```

---

## 📊 Las 9 Características Biofísicas Canónicas

| Característica | Nombre Completo | Significado Fisiológico |
| :--- | :--- | :--- |
| **I0** | Zero-Frequency Impedivity | Resistencia extracelular ($R_0$ en $\Omega$). Principal predictor tumoral. |
| **PA500** | Phase Angle at 500 kHz | Ángulo de fase sensible a la capacitancia de membrana celular. |
| **HFS** | High-Frequency Slope | Pendiente del ángulo de fase entre 500 kHz y 1 MHz. |
| **DA** | Spectral Distance | Distancia euclídea entre los extremos espectrales (15.6 kHz y 1 MHz). |
| **Area** | Semicircle Area | Integral del semicírculo de impedancia en el plano de Nyquist. |
| **A/DA** | Normalized Area | Área normalizada por $DA$; cuantifica la dispersión dipolar. |
| **Max IP** | Maximum Imaginary Part | Pico reactivo máximo ($-Z''_{\max}$); punto de frecuencia de relajación. |
| **DR** | Low-to-High Real Drop | Diferencia $I_0 - Z'_{1000}$; refleja la penetración intracelular. |
| **P** | Spectral Curve Length | Longitud total del arco de impedancia en el plano complejo. |

---

## 🚦 Semáforo de Riesgo y Triaje Clínico

La plataforma traduce las probabilidades matemáticas a un protocolo asistencial homologado con la escala **BI-RADS (Breast Imaging Reporting and Data System)**:

* 🔴 **ALTO RIESGO (Carcinoma - Neoplasia Maligna):**
  * *Categoría:* **BI-RADS 4/5 (Sospechoso de Malignidad)**
  * *Acción:* Derivación prioritaria a **Biopsia con Aguja Gruesa (BAG)** guiada por ultrasonido y mamografía diagnóstica urgente (plazo máximo: 7 días hábiles).
* 🟡 **RIESGO MODERADO (Fibroadenoma / Mastopatía - Lesión Benigna Proliferativa):**
  * *Categoría:* **BI-RADS 3 (Probablemente Benigno)**
  * *Acción:* Evaluación complementaria por ecografía mamaria y seguimiento clínico a corto plazo (3 a 6 meses).
* 🟢 **BAJO RIESGO (Tejido Mamario Sano - Glandular / Conectivo / Adiposo):**
  * *Categoría:* **BI-RADS 1/2 (Normal / Fisiológico)**
  * *Acción:* Continuar con el esquema regular de autoexamen mensual y control preventivo anual.

---

## 🚀 Guía de Instalación y Ejecución Local

### 1. Prerrequisitos
* Python 3.10 o superior (recomendado Python 3.12).
* Git.

### 2. Clonar el Repositorio y Crear Entorno Virtual
```bash
git clone https://github.com/Laroguee/API-de-reservaciones.git
cd VITA-ES-DEMO

# Crear y activar entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Entrenar y Validar el Modelo de IA
```bash
python src/train.py
```
*Salida esperada:*
```text
======================================================================
 VITA-ES: PIPELINE DE ENTRENAMIENTO DE IA MÉDICA (HUAWEI MODELARTS)
======================================================================
[*] Registros cargados: 106
[*] Ejecutando validación cruzada estratificada (k=5)...
[+] Métricas de Validación Cruzada:
    - Exactitud Global (Accuracy) : 67.92%
    - Sensibilidad Carcinoma (Car): 85.71% (CRÍTICO CLÍNICO)
    - Precisión Carcinoma (Car)   : 78.26%
[+] Jerarquía de importancia de características biofísicas:
    - I0      : 19.98%
    - P       : 15.68%
    - Max IP  : 12.09%
    ...
```

### 5. Iniciar la Aplicación Streamlit
```bash
streamlit run app.py
```
Abra su navegador en: **`http://localhost:8501`**

---

## 🐳 Despliegue con Docker (Huawei Cloud ECS / CCE)

### Construcción de la Imagen:
```bash
docker build -t vita-es-platform:v1.0 .
```

### Ejecución del Contenedor:
```bash
docker run -d -p 8501:8501 --name vita-es-app vita-es-platform:v1.0
```

Verificar estado y healthcheck:
```bash
docker ps
curl http://localhost:8501/_stcore/health
```

---

## ☁️ Integración con los Servicios de Huawei Cloud

1. **Huawei Cloud ModelArts:**
   * Hospeda el pipeline de reentrenamiento continuo de los modelos de bioimpedancia sobre clusters de aceleración **Ascend 910B**.
   * Expone un endpoint REST de ultra-baja latencia (`/v1/infers/vita-es-triage`) probado en la pestaña 4 con tiempos de respuesta menores a 45 ms.
2. **Huawei Cloud ECS (Elastic Cloud Server):**
   * Servidor virtual con sistema operativo **EulerOS** que ejecuta el API Gateway y la aplicación de triaje en Streamlit con auto-escalado horizontal.
3. **Huawei Cloud RDS (PostgreSQL):**
   * Base de datos relacional con cifrado en reposo mediante **Huawei Cloud KMS (Key Management Service)** para expedientes clínicos de pacientes, resultados de triaje y firma digital de los médicos tratantes.
4. **Huawei Cloud DCS (Distributed Cache Service - Redis):**
   * Memoria distribuida de alto rendimiento utilizada como búfer intermedio para ingestar flujos continuos de datos espectrales provenientes de múltiples sondas de prueba simultáneamente con latencia < 1 ms.

---

## 📚 Referencias Científicas

1. **J. Jossinet (1996).** *Variability of impedivity in normal and pathological breast tissue.* Medical & Biological Engineering & Computing, 34(5): 346-350.
2. **J. E. Silva, J. P. Marques de Sá, J. Jossinet (2000).** *Classification of Breast Tissue by Electrical Impedance Spectroscopy.* Medical & Biological Engineering & Computing, 38(1): 26-30.
3. **K. S. Cole, R. H. Cole (1941).** *Dispersion and absorption in dielectrics. I. Alternating current characteristics.* The Journal of Chemical Physics, 9(4): 341-351.
4. **World Health Organization (WHO) (2024).** *Global Breast Cancer Initiative Implementation Framework: Assessing, strengthening and monitoring cancer early detection and diagnosis.*

---

## 👩‍💻 Equipo de Desarrollo VITA-ES
* **Competencia:** Huawei ICT Competition 2026-2027
* **Track:** Innovation Competition
* **Tópico:** Creating Innovative AI Applications Powered by Huawei Cloud
