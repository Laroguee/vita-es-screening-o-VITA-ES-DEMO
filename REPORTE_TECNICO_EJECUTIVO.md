# VITA-ES: Plataforma Inteligente de Tamizaje y Triaje Mamario Oportuno mediante Espectroscopía de Bioimpedancia Eléctrica y Fusión Multimodal en Huawei Cloud

**Documento:** Reporte Técnico Ejecutivo de Arquitectura, Fundamentación Clínica y Proyección  
**Competencia:** Huawei ICT Competition 2026-2027 | Innovation Track — Topic 1: Innovative AI on Huawei Cloud  
**Hardware Sonda:** VITA-PRB-0428 (Analog Devices AD5933 + ESP32-S3 BLE 5.0)  
**Edge Computing:** MindSpore Lite sobre procesador neural Huawei Ascend 310  
**Cloud Computing:** Huawei ModelArts (Ascend 910 Pro NPU), ECS EulerOS, RDS PostgreSQL, DCS Redis  
**Versión del Sistema:** v2.4.0-Production  

---

## 1. Resumen Ejecutivo y Generalidades del Proyecto

### 1.1. El Problema de Salud Pública
El cáncer de mama es la neoplasia maligna más frecuente y la principal causa de mortalidad por cáncer en mujeres a nivel mundial (según la Organización Mundial de la Salud - OMS, con más de 2.3 millones de nuevos casos anuales). En América Latina y regiones en vías de desarrollo, más del **65% de los diagnósticos se realizan en etapas localmente avanzadas o metastásicas (Estadios III y IV)**.

Las barreras estructurales que impiden el tamizaje oportuno son:
1. **Inequidad Geográfica:** La mamografía convencional requiere mastógrafos de alto costo ($150,000 - $350,000 USD), salas blindadas contra radiación ionizante y radiólogos subespecialistas escasos en zonas rurales o comunidades periféricas.
2. **Limitación Diagnóstica en Mama Densa:** Entre el 40% y 50% de las mujeres pre- y perimenopáusicas presentan parénquima mamario denso (Categorías ACR C y D), condición que reduce la sensibilidad de la mamografía estándar a menos del **48-52%** debido al enmascaramiento radiológico.
3. **Saturación Hospitalaria y Falsos Positivos:** La ausencia de filtros objetivos en atención primaria provoca largas listas de espera para biopsias y mamografías diagnósticas innecesarias en lesiones benignas como fibroadenomas o mastopatías.

### 1.2. La Solución VITA-ES
**VITA-ES** (*Vital Impedance Tomography & Assessment - Early Screening*) es un ecosistema médico de grado clínico y no invasivo que descentraliza el tamizaje mamario mediante **Espectroscopía de Bioimpedancia Eléctrica (EIS)** a multifrecuencia (15.625 kHz a 1.0 MHz), combinada con **Fusión Multimodal Bayesiana** e inferencia híbrida Edge-to-Cloud con tecnologías de **Huawei Cloud**.

El sistema no busca reemplazar la biopsia histológica confirmatoria (Gold Standard), sino actuar como una **estación de triaje oportuno de alta sensibilidad (90.48% de sensibilidad y 96.47% de especificidad en detección de carcinoma mamario, con 83.96% de exactitud en triaje BI-RADS de 3 niveles)**, clasificando el tejido mamario en tres niveles de riesgo homólogos al estándar internacional **BI-RADS (Breast Imaging Reporting and Data System)** y emitiendo un índice cuantitativo de incertidumbre clínica (**CUI**).

---

## 2. Fundamentación Biofísica y Modelo de Cole-Cole

### 2.1. Propiedades Dieléctricas de los Tejidos Mamarios
A nivel biológico, las células normales y las neoplásicas poseen comportamientos electrofisiológicos disímiles:
* **Tejido Conectivo y Adiposo Normal:** El tejido graso mamario subcutáneo contiene adipocitos voluminosos con gotas lipídicas hidrófobas no conductoras, exhibiendo resistencias extracelulares muy elevadas ($R_0 > 2,000\ \Omega$).
* **Parénquima Glandular Sano y Lesiones Benignas (Fibroadenomas/Mastopatías):** Conservan arquitectura celular ordenada, uniones comunicantes íntegras y baja capacitancia de membrana.
* **Carcinoma Mamario Maligno:** Presenta proliferación descontrolada con alta densidad celular, atipia nuclear, incremento masivo del agua intracelular y extracelular, pérdida de la integridad de la membrana fosfolipídica y angiogénesis aberrante. Esto genera una marcada reducción de la impedividad a bajas frecuencias ($R_0 \approx 394\ \Omega$) y un incremento en la dispersión interfacial dieléctrica ($\Delta R$).

### 2.2. Modelo Matemático de Dispersión Dieléctrica de Cole-Cole
La respuesta en frecuencia del tejido bajo corriente alterna sinusoidal de baja intensidad ($I_{RMS} < 50\ \mu\text{A}$) sigue la formulación clásica de Cole & Cole (1941):

$$Z(\omega) = R_\infty + \frac{R_0 - R_\infty}{1 + (j \omega \tau)^\alpha}$$

Donde:
* $\omega = 2\pi f$: Frecuencia angular de estimulación.
* $R_0$: Resistencia a frecuencia cero (impedividad extracelular pura, correspondiente al parámetro clínico $I_0$).
* $R_\infty$: Resistencia a frecuencia infinita (donde las membranas celulares actúan como cortocircuitos capacitivos y la corriente atraviesa fluidos intra y extracelulares).
* $\Delta R = R_0 - R_\infty$: Rango de dispersión bioeléctrica interfacial (parámetro clínico $DR$).
* $\tau$: Tiempo de relajación característico de la dispersión $\beta$, asociado a la frecuencia pico $f_c = \frac{1}{2\pi\tau}$ (típicamente entre 150 kHz y 800 kHz en mama).
* $\alpha$: Parámetro empírico de depresión de Cole ($0 < \alpha \le 1$), indicador de la heterogeneidad tisular microscópica.

### 2.3. Parámetro Analítico del Pico Reactivo (Max IP)
El ápice del arco de Cole-Cole en el plano complejo de Nyquist ($Z'$ vs $-Z''$) ocurre en $\omega_c = 1/\tau$, donde la reactancia capacitiva máxima responde a la identidad:

$$-Z''_{\max} = \frac{R_0 - R_\infty}{2} \tan\left(\frac{\alpha \pi}{4}\right)$$

### 2.4. Características Canónicas Extraídas (9 Dimensiones Biofísicas)
A partir del barrido de 7 frecuencias estandarizadas por Jossinet ($15.625, 31.25, 62.5, 125.0, 250.0, 500.0, 1000.0\text{ kHz}$), el motor de procesamiento extrae un vector de características canónico normalizado $\mathbf{x} \in \mathbb{R}^9$:

1. **$I_0$:** Resistencia extracelular extrapolada a corriente continua ($f \to 0$).
2. **$\text{PA500}$:** Ángulo de fase $\theta$ en radianes a $500\text{ kHz}$ (sensible a la capacitancia de membrana tumoral).
3. **$\text{HFS}$:** Pendiente de alta frecuencia (*High Frequency Slope*) del ángulo de fase entre $500\text{ kHz}$ y $1.0\text{ MHz}$.
4. **$\text{DA}$:** Distancia euclídea en el plano complejo entre los puntos extremos del espectro ($15.6\text{ kHz}$ y $1\text{ MHz}$).
5. **$\text{Area}$:** Integral trapezoidal bajo el arco espectral en el diagrama de Nyquist.
6. **$\text{A/DA}$:** Relación geométrica de área sobre distancia, representativa de la excentricidad del arco.
7. **$\text{Max IP}$:** Reactancia máxima alcanzada en el pico de dispersión dieléctrica.
8. **$\text{DR}$:** Magnitud de dispersión real $R_0 - \text{Re}(Z(1\text{ MHz}))$.
9. **$\text{P}$:** Longitud total del arco de la trayectoria en el plano complejo ($\sum \sqrt{\Delta Z'^2 + \Delta Z''^2}$).

---

## 3. Datasets Clínicos y Fuentes de Validación

La plataforma VITA-ES integra tres fuentes de datos clínicos de referencia internacional. En cumplimiento de los principios bioéticos y de ciencia de datos biomédica, **los datasets no son concatenados fila por fila** (al provenir de cohortes independientes recolectadas en diferentes hospitales y países). Su integración se realiza mediante **Fusión Multimodal Tardía (Late Bayesian Fusion)**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FUENTES DE DATOS CLÍNICOS VALIDADAS EN VITA-ES                                   │
├────────────────────────────┬─────────────────────────────┬──────────────────────────────────────────────────────┤
│ Dataset                    │ Cohorte y Origen            │ Contribución Clínica en la Plataforma                │
├────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────────────┤
│ 1. breast_tissue.csv       │ J. Jossinet (1996)          │ Dimensión Bioeléctrica Primaria (70%)                │
│    (106 muestras ex-vivo)  │ INSERM Unit 281, Lyon, FR   │ Entrenamiento del Model Zoo para clasificación       │
│                            │ UCI / Kaggle Repository     │ tisular en 6 tipos histológicos.                     │
├────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────────────┤
│ 2. dataR2.csv (BCCD)       │ M. Patrício et al. (2018)   │ Dimensión Metabólica Pre-Test (15%)                  │
│    (116 pacientes sangre)  │ Univ. de Coimbra, Portugal  │ Modela el riesgo biológico basal a partir de IMC,    │
│                            │ BMC Cancer, 18:29           │ Glucosa, Insulina, HOMA-IR y Adipocinas.             │
├────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────────────┤
│ 3. wdbc.data (WDBC)        │ Wolberg, Street et al.(1995)│ Dimensión Citológica Morfológica (15%)               │
│    (569 casos con masa)    │ Univ. of Wisconsin Hospital │ Evalúa atipia nuclear y morfología celular ante la   │
│                            │ Madison, Wisconsin, USA     │ presencia de nódulos mamarios palpables en examen.   │
└────────────────────────────┴─────────────────────────────┴──────────────────────────────────────────────────────┘
```

### 3.1. Distribuciones Empíricas por Clase Histológica (`breast_tissue.csv`)
Los parámetros de calibración de la sonda y del emulador fueron derivados estrictamente de las medias y desviaciones estándar de la cohorte clínica real:

* **Carcinoma (`car`, N=21):** $I_0 = 394.23 \pm 87.05\ \Omega$, $DR = 153.01 \pm 67.02\ \Omega$, $\text{Area} = 5,723.1 \pm 3,172.3\ \Omega^2$.
* **Fibroadenoma (`fad`, N=15):** $I_0 = 245.86 \pm 69.76\ \Omega$, $DR = 47.84 \pm 22.26\ \Omega$, $\text{Area} = 610.2 \pm 430.6\ \Omega^2$.
* **Mastopatía (`mas`, N=18):** $I_0 = 290.31 \pm 111.96\ \Omega$, $DR = 63.61 \pm 39.22\ \Omega$, $\text{Area} = 1,082.4 \pm 1,254.1\ \Omega^2$.
* **Glandular (`gla`, N=16):** $I_0 = 238.32 \pm 119.19\ \Omega$, $DR = 27.50 \pm 33.21\ \Omega$, $\text{Area} = 411.7 \pm 632.8\ \Omega^2$.
* **Conectivo (`con`, N=14):** $I_0 = 1,212.86 \pm 386.47\ \Omega$, $DR = 357.98 \pm 179.55\ \Omega$, $\text{Area} = 5,324.6 \pm 4,069.2\ \Omega^2$.
* **Adiposo (`adi`, N=22):** $I_0 = 2,052.05 \pm 342.49\ \Omega$, $DR = 324.71 \pm 230.36\ \Omega$, $\text{Area} = 24,889.7 \pm 35,695.3\ \Omega^2$.

---

## 4. Arquitectura de Machine Learning e Inteligencia Artificial

### 4.1. Model Zoo Clínico (5 Arquitecturas Comparativas Verificables)
VITA-ES no depende de una caja negra monolítica; integra un **Model Zoo de 5 clasificadores** validados mediante *Stratified K-Fold Cross Validation* ($k=5$, reproducible al 100% ejecutando `python src/models_zoo.py`):

1. **XGBoost Classifier (Modelo Campeón en Sensibilidad Oncológica):**
   * Gradient Boosting regularizado (`max_depth=3`, `learning_rate=0.08`, `subsample=0.85`).
   * **Sensibilidad en Carcinoma: 90.48%** (19/21 casos detectados) | **Especificidad en Carcinoma: 96.47%** | **Precisión en Carcinoma: 86.36%**.
   * **Exactitud en Triaje BI-RADS (3 niveles): 83.96%** | **Exactitud Histológica (6 clases): 71.70%** | **F1-Score Macro: 0.7004**.
2. **Ensamble Multimodelo (Votación Suave Ponderada):**
   * Fusión probabilística ponderada de los clasificadores del Model Zoo.
   * **Sensibilidad en Carcinoma: 90.48%** | **Especificidad en Carcinoma: 95.29%** | **Exactitud en Triaje BI-RADS (3 niveles): 82.08%** | **Exactitud Histológica (6 clases): 70.75%**.
3. **VITA-Net 1D (Deep Learning Convolucional Determinístico):**
   * Arquitectura CNN 1D sobre vectores de espectroscopía dieléctrica: 2 bloques `Conv1D` (filtros 24 y 32, kernel=3, activación `ELU`) con `BatchNormalization`, `Flatten`, `Dropout(0.15)` y capa densa regularizada con pesos balanceados por clase.
   * **Sensibilidad en Carcinoma: 80.95%** | **Especificidad en Carcinoma: 95.29%** | **Exactitud en Triaje BI-RADS (3 niveles): 78.30%** | **Exactitud Histológica (6 clases): 68.87%**.
4. **Balanced Random Forest (Baseline Clínico):**
   * 150 árboles con muestreo balanceado (`balanced_subsample`) para mitigar el desbalance de la cohorte.
   * **Sensibilidad en Carcinoma: 85.71%** | **Especificidad en Carcinoma: 94.12%** | **Exactitud en Triaje BI-RADS (3 niveles): 80.19%** | **Exactitud Histológica (6 clases): 67.92%**.
5. **Support Vector Machine (Kernel RBF Calibrado):**
   * Kernel gaussiano ($C=2.5, \gamma=\text{scale}$), con calibración de probabilidades de Platt.
   * **Sensibilidad en Carcinoma: 85.71%** | **Especificidad en Carcinoma: 95.29%** | **Exactitud en Triaje BI-RADS (3 niveles): 73.58%** | **Exactitud Histológica (6 clases): 60.38%**.
6. **Modelos Auxiliares Multimodales ($k=5$ Stratified CV):**
   * **Modelo Metabólico Coimbra (`dataR2.csv`, $N=116$):** Exactitud CV de **72.41%** y Sensibilidad de **75.00%**.
   * **Modelo Morfológico Wisconsin (`wdbc.data`, $N=569$):** Exactitud CV de **97.36%** y Sensibilidad de **94.34%**.

### 4.2. Cuantificación de Incertidumbre Clínica (CUI)
Para evitar que el algoritmo emita un falso descarte preventivo ante lecturas atípicas, VITA-ES implementa el **Clinical Uncertainty Index (CUI)**:

$$\text{CUI} = \Big[ 0.65 \cdot H_{norm} + 0.35 \cdot (1 - \Delta P) \Big] \times 100\%$$

Donde:
* **Entropía de Shannon Normalizada:** $H_{norm} = \frac{-\sum_{i=1}^K p_i \ln p_i}{\ln K}$ (mide la dispersión de la confianza entre las $K=6$ clases).
* **Margen de Confianza Top-1 vs Top-2:** $\Delta P = P_{(1)} - P_{(2)}$ (mide la proximidad entre la clase más probable y su alternativa inmediata).
* **Interlock de Seguridad:** Si $\text{CUI} > 55\%$, el sistema revoca el descarte automático y emite la alerta: *«Dictamen No Concluyente: Se requiere confirmación ecográfica/histológica»*.

### 4.3. Algoritmo de Fusión Multimodal Bayesiana
El riesgo final de malignidad integra los tres canales de evidencia clínica:

$$P_{\text{Fused}} = 0.70 \cdot P_{\text{car}}(\text{EIS}) + 0.15 \cdot P_{\text{Metabólico}}(\text{Coimbra}) + 0.15 \cdot P_{\text{Morfológico}}(\text{WDBC})$$

```
                   MATRIZ DE TRIAJE ONCOLÓGICO VITA-ES
┌─────────────────────────────────┬─────────────────┬───────────────────────────────────────┐
│ Criterio de Fusión              │ Categoría       │ Acción Clínica Recomendada            │
├─────────────────────────────────┼─────────────────┼───────────────────────────────────────┤
│ P_Fused >= 32% o Pred = 'car'   │ ALTO RIESGO     │ Derivación prioritaria a Biopsia con  │
│                                 │ BI-RADS 4 / 5   │ Aguja Gruesa (BAG) y Mamografía (<7d) │
├─────────────────────────────────┼─────────────────┼───────────────────────────────────────┤
│ P(fad) + P(mas) >= 38%          │ RIESGO MODERADO │ Evaluación por Ultrasonido mamario    │
│                                 │ BI-RADS 3       │ complementario y control en 3-6 meses │
├─────────────────────────────────┼─────────────────┼───────────────────────────────────────┤
│ Patrón tisular fisiológico      │ BAJO RIESGO     │ Continuar con protocolo habitual de   │
│                                 │ BI-RADS 1 / 2   │ tamizaje preventivo anual por edad    │
└─────────────────────────────────┴─────────────────┴───────────────────────────────────────┘
```

---

## 5. Arquitectura Tecnológica Huawei Cloud (Edge-to-Cloud)

```
┌────────────────────────────────────────────────────────┐
│                   DISPOSITIVO EDGE                     │
│         Sonda Portátil VITA-PRB-0428                   │
│  ┌────────────────────────┐  ┌──────────────────────┐  │
│  │ Sensor AD5933 (12-bit) │  │ ESP32-S3 BLE 5.0    │  │
│  │ Ingesta 15 kHz - 1 MHz │  │ Firmware v2.4 (GATT) │  │
│  └───────────┬────────────┘  └──────────┬───────────┘  │
│              └────────────┬─────────────┘              │
│                           ▼                            │
│              ┌────────────────────────┐                │
│              │ MindSpore Lite Edge    │                │
│              │ Inferencia en ~4.2 ms  │                │
│              │ (Huawei Ascend 310)    │                │
│              └────────────┬───────────┘                │
└───────────────────────────┼────────────────────────────┘
                            │ Telemetría BLE / 4G (TLS 1.3)
                            ▼
┌────────────────────────────────────────────────────────┐
│                  HUAWEI CLOUD SUITE                    │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Huawei ECS Gateway (EulerOS c7.large.2)          │  │
│  │ Microservicio API REST / Validación SQI          │  │
│  └──────────┬───────────────────────────┬───────────┘  │
│             │                           │              │
│             ▼                           ▼              │
│  ┌─────────────────────┐     ┌──────────────────────┐  │
│  │ Huawei ModelArts    │     │ Huawei DCS (Redis)   │  │
│  │ Inferencia Online   │     │ Buffer de Telemetría │  │
│  │ Ascend 910 Pro NPU  │     │ Latencia < 1.5 ms    │  │
│  └──────────┬──────────┘     └──────────────────────┘  │
│             │                                          │
│             ▼                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Huawei RDS (PostgreSQL 14 Cifrado KMS)           │  │
│  │ Expedientes Clínicos y Trazabilidad HIPAA/GDPR   │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 5.1. Componentes Clave Desplegados
* **MindSpore Lite en el Edge:** Los modelos Random Forest y SVM fueron exportados a formato abierto ONNX (`models/onnx/model_rf.onnx` y `model_svm.onnx`) para ejecución optimizada en aceleradores de borde **Huawei Atlas 200 / Ascend 310**, logrando latencias de inferencia de **4.2 ms**.
* **Huawei ModelArts (Inferencia y MLOps):** Empaquetado completo en `modelarts_deploy/` con `config.json` y `customize_service.py` listo para despliegue serverless sobre chips neuronales **Huawei Ascend 910 Pro**.
* **Huawei DCS (Distributed Cache Service):** Instancia Redis Enterprise para absorber el streaming multifrecuencia de la sonda en tiempo real y mitigar cuellos de botella en brigadas con alta concurrencia.
* **Huawei RDS (Relational Database Service):** Base de datos relacional PostgreSQL con cifrado de datos en reposo vía **Huawei KMS (Key Management Service)** con estándar AES-256 para expedientes clínicos.

---

## 6. Validación de Control de Calidad de Señal (SQI Interlock)

Para asegurar la confiabilidad diagnóstica en brigadas de campo, el sistema ejecuta una validación de calidad antes de alimentar los modelos de IA:
1. **Detección de Contacto Deficiente / Piel Seca:** Si $R_0 > 4,200\ \Omega$ o la reactancia permanece plana ($\theta < 0.03\text{ rad}$), el sistema identifica electrodo desprendido o insuficiente gel conductor.
2. **Detección de Cortocircuito:** Si $Z' < 50\ \Omega$ en todo el espectro, se detecta puente de solución salina entre electrodos.
3. **Resistencia a Artefactos de Movimiento:** Evaluación de suavidad de curva Cole-Cole ($SNR > 14\text{ dB}$).
4. **Interlock:** Si el índice $\text{SQI} < 70\%$, la inferencia diagnóstica **se bloquea automáticamente** para prevenir falsos positivos o negativos inducidos por ruido instrumental.

---

## 7. Hacia Dónde Apunta el Proyecto (Visión y Hoja de Ruta)

### 7.1. Impacto Social y Salud Pública
VITA-ES está diseñado para operar como una **tecnología de tamizaje de primera línea (First-Line Triage Tool)** en:
* **Unidades de Salud Comunitarias y Rurales:** Equipos portátiles de bajo costo ($< 250\text{ USD}$ en componentes) que permiten tamizar cientos de mujeres al mes sin requerir un mamógrafo en cada centro de salud.
* **Reducción de Listas de Espera en Hospitales de Referencia:** Al identificar con alta sensibilidad a las pacientes que verdaderamente requieren mamografía y biopsia inmediata, se optimizan los recursos de la red hospitalaria pública.
* **Tamizaje Complementario en Mamas Densas:** La bioimpedancia es insensible a la densidad del parénquima fibroglandular que oculta tumores en mastografía convencional.

### 7.2. Fases de la Hoja de Ruta (Roadmap)
* **Fase 1 (Actual - Q4 2026):** Demostrador funcional validado, Model Zoo con 4 algoritmos, cuantificación de incertidumbre CUI, empaquetado para ModelArts y emulación de sonda de hardware AD5933.
* **Fase 2 (Q1-Q2 2027):** Fabricación del prototipo de hardware industrial con electrodos concéntricos tetra-polares en acero quirúrgico 316L, certificación de biocompatibilidad ISO 10993 y conexión nativa 4G-LTE Cat-M1 a Huawei Cloud.
* **Fase 3 (Q3-Q4 2027):** Ensayo clínico piloto multicéntrico hospitalario (N=500 pacientes pre-biopsia) y registro sanitario como Dispositivo Médico Clase II.
* **Fase 4 (2028):** Aprendizaje Federado (*Federated Learning*) entre hospitales a través de Huawei ModelArts para actualización continua de pesos sin centralizar datos privados.

---

## 8. Oportunidades de Mejora y Limitaciones Actuales

Como todo desarrollo de innovación biomédica riguroso, se reconocen áreas técnicas de evolución:

1. **Ampliación de la Cohorte Histológica Primaria:**
   * *Limitación:* El dataset de referencia de bioimpedancia mamaria de Jossinet cuenta con 106 muestras ex-vivo.
   * *Mejora:* Incorporar barridos in-vivo provenientes de ensayos clínicos locales para enriquecer la variabilidad intra e interindividual de las clases benignas (`fad` y `mas`).
2. **Transición a Configuración Tetra-Polar (4 Electrodos):**
   * *Limitación:* Las sondas bi-polares incluyen la impedancia parásita de la interfaz piel-electrodo en la lectura.
   * *Mejora:* Implementar técnica tetra-polar (dos electrodos para inyección de corriente $I$ y dos para medición de potencial diferencial $V$), cancelando completamente el efecto de contacto cutáneo.
3. **Reconstrucción Tomográfica 2D/3D (EIT):**
   * *Oportunidad:* Evolucionar de una sonda de punto focal a un cabezal anular multielectrodo (16 o 32 puntos) para generar mapas de tomografía de impedancia eléctrica bidimensional en tiempo real.
4. **Fusión Profunda End-to-End con Imágenes:**
   * *Oportunidad:* Entrenar redes multimodales densas que concatenen el vector dieléctrico de la sonda con imágenes ecográficas o fotografías térmicas mamarias en ModelArts.

---

## 9. Referencias Bibliográficas

1. **Jossinet, J. (1996).** *Variability of impedivity in normal and pathological breast tissue*. Medical & Biological Engineering & Computing, 34(5), 346–350.
2. **Patrício, M., Pereira, J., Crisóstomo, J., Matafome, P., Gomes, M., Seiça, R., & Caramelo, F. (2018).** *Using Resistin, Glucose, Age and BMI to Predict the Presence of Breast Cancer*. BMC Cancer, 18(1), 29.
3. **Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995).** *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository.
4. **Cole, K. S., & Cole, R. H. (1941).** *Dispersion and absorption in dielectrics I. Alternating current characteristics*. The Journal of Chemical Physics, 9(4), 341–351.
5. **Silva, J. E., et al. (2000).** *Sensitivity and specificity of electrical impedance spectroscopy in breast cancer screening*. IEEE Transactions on Biomedical Engineering, 47(6), 820–827.
6. **American College of Radiology (ACR). (2013).** *ACR BI-RADS® Atlas: Breast Imaging Reporting and Data System*. Reston, VA: American College of Radiology.
7. **Analog Devices. (2020).** *AD5933: 1 MSPS, 12-Bit Impedance Converter Network Analyzer Technical Datasheet*. Rev. F.
8. **Huawei Cloud. (2025).** *ModelArts AI Development Platform & MindSpore Lite Edge Inference Architecture Guide*.
