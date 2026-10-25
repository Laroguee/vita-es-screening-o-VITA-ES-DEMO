"""
Script de Verificación Local del Paquete de Despliegue Huawei Cloud ModelArts
Proyecto VITA-ES | Huawei ICT Competition 2026-2027

Simula una llamada HTTP a handle() en customize_service.py para validar
que el paquete está 100% listo para subirse a la consola de ModelArts.
"""

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Importar servicio local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from customize_service import handle

def test_modelarts_service():
    print("=" * 70)
    print(" VITA-ES: TEST LOCAL DE SERVICIO HUAWEI CLOUD MODELARTS")
    print("=" * 70)

    # Payload de prueba representativo de carcinoma
    sample_payload = {
        "patient_id": "PAT-HUAWEI-DEMO-001",
        "model_name": "XGBoost",
        "features": {
            "I0": 425.0,
            "PA500": 0.142,
            "HFS": 0.034,
            "DA": 182.0,
            "Area": 8450.0,
            "A/DA": 46.4,
            "Max IP": 74.0,
            "DR": 172.0,
            "P": 525.0
        },
        "metabolic_factors": {
            "age": 52,
            "bmi": 26.4,
            "glucose": 98.0
        }
    }

    print("[*] Enviando petición a handle(data)...")
    response = handle(sample_payload)

    print("\n[+] Respuesta recibida de ModelArts Service:")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print("=" * 70)

if __name__ == "__main__":
    test_modelarts_service()
