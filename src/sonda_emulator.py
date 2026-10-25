"""
Emulador de Telemetría de Hardware: Sonda Biomédica Portátil VITA-ES
Proyecto VITA-ES | Huawei ICT Competition 2026-2027

Emula el comportamiento bioeléctrico y físico del hardware VITA-PRB-0428:
- Front-End Analógico: Analog Devices AD5933 (Convertidor de Impedancia de 12 bits)
- Microcontrolador: ESP32-S3 (Dual Core con soporte MindSpore Lite Micro)
- Electrodos: Anillos concéntricos diferenciales Ag/AgCl
- Protocolos de Telemetría: Bluetooth Low Energy (BLE 5.0) y JSON Streaming over Wi-Fi
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

# Imports del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import CLASS_MAPPING, STANDARD_FREQUENCIES_KHZ, generate_cole_sweep
from feature_extraction import assess_signal_quality, extract_features_from_sweep


class VITAProbeEmulator:
    """
    Emulador de instrumental biomédico para la sonda de mano VITA-ES (AD5933).
    Genera tramas de telemetría médica en tiempo real incluyendo:
    - Estado de batería y consumo térmico
    - Impedancia de contacto piel-electrodo
    - Barrido espectroscópico multifrecuencia (15.6 kHz a 1 MHz)
    - Tramas hexadecimales crudas de paquetes BLE (Bluetooth GATT Characteristic)
    """

    DEVICE_ID = "VITA-PRB-0428"
    FIRMWARE_VERSION = "v2.4.1-MindSpore-Lite"
    HARDWARE_REV = "HW-AD5933-RevC"

    def __init__(self, data_stream_path: str = "data/telemetry_stream.json"):
        self.data_stream_path = data_stream_path
        self.battery_pct = 94.0
        self.temperature_c = 36.6

    def generate_ble_hex_payload(self, r0: float, r_inf: float, sqi: float) -> str:
        """
        Genera una trama binaria simulada formateada en hexadecimal representativa del protocolo BLE.
        Header (0xAA55) + DeviceID + Status + R0 + Rinf + SQI + CRC16
        """
        r0_int = int(np.clip(r0, 0, 65535))
        rinf_int = int(np.clip(r_inf, 0, 65535))
        sqi_byte = int(np.clip(sqi, 0, 100))
        batt_byte = int(np.clip(self.battery_pct, 0, 100))
        
        # Hex bytes representativos
        hex_tokens = [
            "AA", "55",  # Preamble de sincronización médica
            "04", "28",  # ID Sonda (0428)
            f"{batt_byte:02X}",  # Batería
            f"{sqi_byte:02X}",   # Calidad SQI
            f"{(r0_int >> 8) & 0xFF:02X}", f"{r0_int & 0xFF:02X}",      # R0 en 2 bytes
            f"{(rinf_int >> 8) & 0xFF:02X}", f"{rinf_int & 0xFF:02X}",  # R_inf en 2 bytes
            "DE", "AD"   # CRC-16 Checksum
        ]
        return " ".join(hex_tokens)

    def acquire_telemetry_scan(
        self,
        tissue_class: str = "car",
        noise_level: float = 0.015,
        simulate_faulty_contact: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta un ciclo de escaneo bioeléctrico completo en la sonda:
        1. Modula la resistencia tisular según la patología emulada.
        2. Inyecta variaciones estocásticas de impedancia de contacto piel-electrodo.
        3. Evalúa el Signal Quality Index (SQI).
        4. Empaqueta la telemetría en JSON y genera la trama BLE hexadecimal.
        """
        # Desgaste gradual de batería por escaneo
        self.battery_pct = max(5.0, self.battery_pct - np.random.uniform(0.05, 0.15))
        # Temperatura del cabezal térmico del sensor
        self.temperature_c = round(36.5 + np.random.normal(0, 0.2), 1)

        info = CLASS_MAPPING.get(tissue_class, CLASS_MAPPING["car"])
        params = info["cole_params"]

        if simulate_faulty_contact:
            # Simular electrodo desprendido o piel seca (R0 > 6000 Ohm)
            r0 = 6800.0 + np.random.uniform(-300, 500)
            r_inf = 1200.0
            tau = 5.0e-6
            alpha = 0.40
            noise_std = 0.12
            electrode_status = "FAULTY_CONTACT_HIGH_IMPEDANCE"
        else:
            var = 1.0 + np.random.uniform(-0.05, 0.05)
            r0 = params["R0"] * var
            r_inf = params["Rinf"] * var
            tau = params["tau"] * (1.0 + np.random.uniform(-0.04, 0.04))
            alpha = min(0.95, max(0.60, params["alpha"] * (1.0 + np.random.uniform(-0.02, 0.02))))
            noise_std = noise_level
            electrode_status = "OPTIMAL_DIFFERENTIAL_CONTACT"

        # Generar barrido espectral multifrecuencia
        sweep_df = generate_cole_sweep(
            r0=r0,
            r_inf=r_inf,
            tau=tau,
            alpha=alpha,
            noise_std=noise_std
        )

        # Evaluar SQI
        sqi_res = assess_signal_quality(sweep_df)

        # Extraer características biofísicas
        features = extract_features_from_sweep(sweep_df)

        # Generar payload hexadecimal BLE
        ble_hex = self.generate_ble_hex_payload(r0, r_inf, sqi_res["sqi_score"])

        packet = {
            "device_metadata": {
                "device_id": self.DEVICE_ID,
                "firmware": self.FIRMWARE_VERSION,
                "hardware_rev": self.HARDWARE_REV,
                "battery_pct": round(self.battery_pct, 1),
                "temperature_c": self.temperature_c,
                "electrode_status": electrode_status,
                "sqi_score": sqi_res["sqi_score"],
                "snr_db": sqi_res["snr_db"],
                "is_signal_valid": sqi_res["is_acceptable"],
                "ble_packet_raw_hex": ble_hex
            },
            "tissue_emulated": {
                "code": tissue_class,
                "name": info["name"],
                "expected_risk": info["risk_level"]
            },
            "features_extracted": features,
            "sweep_points": sweep_df.to_dict(orient="records"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Guardar automáticamente en el buffer compartido
        self.save_stream(packet)

        return packet, sweep_df

    def save_stream(self, packet: Dict[str, Any]):
        """Persiste la trama de telemetría en disco para consumo por Streamlit o APIs externas."""
        try:
            os.makedirs(os.path.dirname(self.data_stream_path), exist_ok=True)
            with open(self.data_stream_path, "w", encoding="utf-8") as f:
                json.dump(packet, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Error al escribir telemetría: {e}")

    @classmethod
    def read_latest_stream(cls, stream_path: str = "data/telemetry_stream.json") -> Optional[Tuple[Dict[str, Any], pd.DataFrame]]:
        """Lee la última trama emitida por la sonda emulada."""
        if not os.path.exists(stream_path):
            return None
        try:
            with open(stream_path, "r", encoding="utf-8") as f:
                packet = json.load(f)
            sweep_df = pd.DataFrame(packet.get("sweep_points", []))
            return packet, sweep_df
        except Exception:
            return None


if __name__ == "__main__":
    emulator = VITAProbeEmulator()
    print("=" * 70)
    print(" VITA-ES: EMULADOR DE TELEMETRÍA DE SONDA BIOMÉDICA VITA-PRB-0428")
    print("=" * 70)
    print("[*] Simulando escaneo de tejido de Carcinoma mamario...")
    packet, df = emulator.acquire_telemetry_scan(tissue_class="car", noise_level=0.012)
    meta = packet["device_metadata"]
    print(f"[+] Dispositivo: {meta['device_id']} | Firmware: {meta['firmware']}")
    print(f"[+] Batería: {meta['battery_pct']}% | Temp: {meta['temperature_c']} °C | SQI: {meta['sqi_score']}% (Válido: {meta['is_signal_valid']})")
    print(f"[+] Trama BLE Hexadecimal: {meta['ble_packet_raw_hex']}")
    print(f"[+] Características I0: {packet['features_extracted']['I0']:.1f} Ohm | Max IP: {packet['features_extracted']['Max IP']:.1f} Ohm")
    print(f"[+] Telemetría guardada en '{emulator.data_stream_path}'.")
    print("=" * 70)
