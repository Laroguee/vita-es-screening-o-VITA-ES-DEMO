# ==============================================================================
# VITA-ES: Dockerfile para Despliegue en Huawei Cloud ECS / CCE
# Huawei ICT Competition 2026-2027 | Innovation Track
# ==============================================================================

FROM python:3.12-slim

# Metadatos del contenedor
LABEL maintainer="VITA-ES Engineering Team <contact@vitaes-health.ai>"
LABEL project="VITA-ES Electrical Bioimpedance Triage Platform"
LABEL huawei_cloud_target="ECS c7.large.2 / CCE Kubernetes"

# Variables de entorno para Python y Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Instalar dependencias mínimas del sistema y curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente, modelos y datos
COPY data/ /app/data/
COPY Data/ /app/Data/
COPY models/ /app/models/
COPY src/ /app/src/
COPY app.py /app/app.py

# Crear usuario sin privilegios para cumplimiento de ciberseguridad médica
RUN useradd -m -u 1001 vitaes && chown -R vitaes:vitaes /app
USER vitaes

# Exponer el puerto estándar de la plataforma clínica Streamlit
EXPOSE 8501

# Monitoreo de salud del contenedor en Huawei Cloud CCE / ECS
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicio del servidor clínico VITA-ES
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
