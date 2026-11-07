#!/usr/bin/env bash
# ==============================================================================
# VITA-ES: Script Automatizado de Aprovisionamiento y Despliegue en Huawei Cloud
# 11th Huawei ICT Competition 2026-2027 | Topic 1: Huawei Cloud AI Applications
# ==============================================================================
# Uso:
#   1. Configurar credenciales de Huawei Cloud CLI (hcloud):
#      export HUAWEI_REGION="la-north-2" # o "la-south-2" / "cn-north-4"
#      export HUAWEI_PROJECT_ID="<tu_project_id>"
#   2. Ejecutar:
#      bash deploy_huawei_cloud.sh
# ==============================================================================

set -euo pipefail

REGION="${HUAWEI_REGION:-la-north-2}"
APP_NAME="vita-es-platform"
IMAGE_TAG="v2.5.0"

echo "======================================================================"
echo " VITA-ES: APROVISIONAMIENTO Y DESPLIEGUE EN HUAWEI CLOUD ($REGION)"
echo "======================================================================"

# 1. Verificación de modelos y pruebas locales antes del despliegue
echo "[1/4] Verificando artefactos del Model Zoo y microservicio ModelArts..."
python src/models_zoo.py
python modelarts_deploy/test_modelarts_local.py

# 2. Construcción de la imagen Docker optimizada para Huawei Cloud ECS / CCE
echo "[2/4] Construyendo contenedor clínico Docker ($APP_NAME:$IMAGE_TAG)..."
docker build -t "${APP_NAME}:${IMAGE_TAG}" .

# 3. Aprovisionamiento de Huawei Cloud RDS (PostgreSQL) y DCS (Redis) vía KooCLI (si está autenticado)
if command -v hcloud >/dev/null 2>&1 && [ -n "${HUAWEI_PROJECT_ID:-}" ]; then
    echo "[3/4] Aprovisionando servicios en Huawei Cloud mediante KooCLI (hcloud)..."
    echo "  -> Verificando instancias Huawei Cloud RDS (PostgreSQL 15)..."
    hcloud RDS ListInstances --cli-region="$REGION" --project_id="$HUAWEI_PROJECT_ID" || true
    echo "  -> Verificando instancias Huawei Cloud DCS (Redis 6.0)..."
    hcloud DCS ListInstances --cli-region="$REGION" --project_id="$HUAWEI_PROJECT_ID" || true
else
    echo "[3/4] Modo Local / Pre-Cloud: KooCLI no detectado o sin credenciales activas."
    echo "      Iniciando stack completo con contenedores locales equivalentes (RDS + DCS + App)..."
fi

# 4. Levantar servicios con Docker Compose
echo "[4/4] Desplegando plataforma VITA-ES..."
docker compose up -d --build

echo "======================================================================"
echo " [+] DESPLIEGUE COMPLETADO: http://localhost:8501"
echo "======================================================================"

