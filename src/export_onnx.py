"""
Pipeline de Exportación a Formato Abierto ONNX para Huawei MindSpore Lite y ModelArts
Proyecto VITA-ES | Huawei ICT Competition 2026-2027

Convierte los modelos entrenados a formato ONNX estándar para ejecución acelerada
en procesadores Huawei Ascend 310 (Edge Sonda) y Ascend 910 Pro (Cloud ModelArts).
"""

import os
import sys
import json
import joblib
import numpy as np

# Imports ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Imports locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import FEATURE_NAMES


def export_models_to_onnx(models_dir: str = "models", onnx_output_dir: str = "models/onnx") -> dict:
    """
    Convierte los clasificadores Scikit-Learn a formato ONNX optimizado.
    """
    print("=" * 70)
    print(" VITA-ES: EXPORTADOR DE MODELOS A FORMATO ONNX / HUAWEI MINDSPORE LITE")
    print("=" * 70)

    os.makedirs(onnx_output_dir, exist_ok=True)
    exported = {}

    # Definir tipo de entrada estándar: tensor flotante (Batch, 9)
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]

    # 1. Exportar Random Forest
    rf_path = os.path.join(models_dir, "model_rf.pkl")
    if os.path.exists(rf_path):
        print("[*] Convirtiendo Balanced Random Forest a ONNX...")
        rf_model = joblib.load(rf_path)
        onnx_rf = convert_sklearn(rf_model, initial_types=initial_type, target_opset=14)
        out_rf_path = os.path.join(onnx_output_dir, "model_rf.onnx")
        with open(out_rf_path, "wb") as f:
            f.write(onnx_rf.SerializeToString())
        exported["Random Forest"] = {
            "onnx_file": out_rf_path,
            "size_kb": round(os.path.getsize(out_rf_path) / 1024, 2),
            "target_hardware": "Huawei Ascend 310 / Atlas 200 MindSpore Lite"
        }
        print(f"    -> Guardado en {out_rf_path} ({exported['Random Forest']['size_kb']} KB)")

    # 2. Exportar SVM
    svm_path = os.path.join(models_dir, "model_svm.pkl")
    if os.path.exists(svm_path):
        print("[*] Convirtiendo SVM (Kernel RBF) a ONNX...")
        svm_model = joblib.load(svm_path)
        try:
            onnx_svm = convert_sklearn(svm_model, initial_types=initial_type, target_opset=14)
            out_svm_path = os.path.join(onnx_output_dir, "model_svm.onnx")
            with open(out_svm_path, "wb") as f:
                f.write(onnx_svm.SerializeToString())
            exported["SVM (RBF)"] = {
                "onnx_file": out_svm_path,
                "size_kb": round(os.path.getsize(out_svm_path) / 1024, 2),
                "target_hardware": "Huawei Ascend 310 MindSpore Lite"
            }
            print(f"    -> Guardado en {out_svm_path} ({exported['SVM (RBF)']['size_kb']} KB)")
        except Exception as e:
            print(f"    [!] Aviso en conversión SVM: {e}")

    # Guardar manifiesto de exportación
    manifest_path = os.path.join(onnx_output_dir, "onnx_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Manifiesto guardado en {manifest_path}")
    print("=" * 70)
    return exported


if __name__ == "__main__":
    export_models_to_onnx()
