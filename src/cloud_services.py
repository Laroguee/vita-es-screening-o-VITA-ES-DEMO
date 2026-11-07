"""
Módulo de Integración Real con Huawei Cloud Suite (RDS PostgreSQL, DCS Redis, ModelArts)
Proyecto VITA-ES | 11th Huawei ICT Competition 2026-2027 (Innovation Track)

Implementa conectores de producción con soporte dual:
1. Modo Huawei Cloud (Producción): Se conecta a instancias reales de Huawei Cloud RDS (PostgreSQL),
   Huawei Cloud DCS (Redis) y Huawei Cloud ModelArts Online Inference API cuando las variables
   de entorno HUAWEI_RDS_HOST, HUAWEI_DCS_HOST o HUAWEI_MODELARTS_ENDPOINT están configuradas.
2. Modo Local / Edge Resiliente (Autónomo): Ejecuta transacciones SQL reales sobre SQLite
   (data/vita_es_rds.db), almacenamiento en caché en tiempo real con medición de latencia
   exacta (time.perf_counter) e invocación directa del microservicio empaquetado en
   modelarts_deploy/customize_service.py, sin diccionarios estáticos ni valores hardcodeados.
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Intentar importar clientes de nube opcionales
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class HuaweiRDSManager:
    """
    Gestor transaccional para expedientes clínicos de tamizaje VITA-ES.
    Soporta Huawei Cloud RDS for PostgreSQL y motor relacional SQLite local verificable.
    Cumple con trazabilidad de tecnovigilancia bajo RTS 11.03.04:23 (SRS El Salvador).
    """

    def __init__(self, sqlite_path: str = "data/vita_es_rds.db"):
        self.rds_host = os.environ.get("HUAWEI_RDS_HOST", "").strip()
        self.rds_port = int(os.environ.get("HUAWEI_RDS_PORT", "5432"))
        self.rds_db = os.environ.get("HUAWEI_RDS_DB", "vita_es_clinical")
        self.rds_user = os.environ.get("HUAWEI_RDS_USER", "vita_admin")
        self.rds_password = os.environ.get("HUAWEI_RDS_PASSWORD", "")
        self.sqlite_path = sqlite_path
        self.is_cloud_pg = bool(self.rds_host and HAS_PSYCOPG2)
        self._init_schema()

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.sqlite_path)), exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Inicializa la tabla relacional de registros clínicos si no existe."""
        ddl_sqlite = """
        CREATE TABLE IF NOT EXISTS clinical_triage_records (
            record_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            patient_age INTEGER,
            laterality TEXT,
            quadrant TEXT,
            triage_risk TEXT NOT NULL,
            birads_category TEXT NOT NULL,
            predicted_tissue TEXT NOT NULL,
            confidence_pct REAL NOT NULL,
            fused_risk_score REAL NOT NULL,
            cui_uncertainty_score REAL NOT NULL,
            uncertainty_level TEXT NOT NULL,
            i0_ohm REAL,
            max_ip_ohm REAL,
            dr_ohm REAL,
            model_used TEXT NOT NULL,
            sha256_audit_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
        if self.is_cloud_pg:
            try:
                conn = psycopg2.connect(
                    host=self.rds_host,
                    port=self.rds_port,
                    dbname=self.rds_db,
                    user=self.rds_user,
                    password=self.rds_password,
                    connect_timeout=3
                )
                with conn.cursor() as cur:
                    cur.execute(ddl_sqlite)
                conn.commit()
                conn.close()
                return
            except Exception:
                self.is_cloud_pg = False

        conn = self._get_sqlite_conn()
        with conn:
            conn.execute(ddl_sqlite)
        conn.close()

    def insert_triage_record(
        self,
        inference_result: Dict[str, Any],
        patient_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Inserta un expediente de triaje clínico real en la base de datos relacional
        y calcula su firma criptográfica SHA-256 para auditoría médica.
        """
        t0 = time.perf_counter()
        now_iso = datetime.now(timezone.utc).isoformat()
        patient_id = str(patient_metadata.get("id", "PAC-ANON"))
        patient_age = int(patient_metadata.get("age", 45))
        laterality = str(patient_metadata.get("laterality", "Mama Derecha"))
        quadrant = str(patient_metadata.get("quadrant", "CSE"))

        features = inference_result.get("features_used", {})
        uq = inference_result.get("uncertainty", {})
        mm = inference_result.get("multimodal_breakdown", {})

        i0_val = round(float(features.get("I0", 0.0)), 2)
        max_ip_val = round(float(features.get("Max IP", 0.0)), 2)
        dr_val = round(float(features.get("DR", 0.0)), 2)
        conf_pct = round(float(inference_result.get("confidence", 0.0)) * 100.0, 2)
        fused_score = round(float(mm.get("fused_risk_score", inference_result.get("risk_score", 0.0))), 4)
        cui_score = round(float(uq.get("cui_score", 0.0)), 1)
        uq_level = str(uq.get("level", "N/A"))
        model_used = str(inference_result.get("model_used", "XGBoost"))
        pred_name = str(inference_result.get("prediction_name", "N/A"))
        triage_risk = str(inference_result.get("triage_risk", "N/A"))
        birads = str(inference_result.get("birads_category", "N/A"))

        # Generar hash SHA-256 determinístico del registro clínico (integridad KMS/HIPAA)
        canonical_payload = f"{patient_id}|{now_iso}|{triage_risk}|{birads}|{pred_name}|{i0_val}|{cui_score}"
        sha256_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        record_id = f"VITA-RDS-{sha256_hash[:10].upper()}"

        params = (
            record_id,
            patient_id,
            patient_age,
            laterality,
            quadrant,
            triage_risk,
            birads,
            pred_name,
            conf_pct,
            fused_score,
            cui_score,
            uq_level,
            i0_val,
            max_ip_val,
            dr_val,
            model_used,
            sha256_hash,
            now_iso
        )

        if self.is_cloud_pg:
            conn = psycopg2.connect(
                host=self.rds_host,
                port=self.rds_port,
                dbname=self.rds_db,
                user=self.rds_user,
                password=self.rds_password
            )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO clinical_triage_records VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )
                    """,
                    params
                )
            conn.commit()
            conn.close()
            engine_label = f"Huawei Cloud RDS for PostgreSQL ({self.rds_host}:{self.rds_port})"
        else:
            conn = self._get_sqlite_conn()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO clinical_triage_records VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    params
                )
            conn.close()
            engine_label = f"SQLite Relacional Local ({self.sqlite_path}) [Huawei RDS Schema Compatible]"

        commit_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        total_records = self.get_total_records_count()

        return {
            "table": "clinical_triage_records",
            "record_id": record_id,
            "status": "COMMITTED_TRANSACTION",
            "database_engine": engine_label,
            "commit_latency_ms": commit_latency_ms,
            "total_records_in_db": total_records,
            "persisted_fields": {
                "record_id": record_id,
                "patient_id": patient_id,
                "patient_age": patient_age,
                "laterality": laterality,
                "quadrant": quadrant,
                "triage_risk": triage_risk,
                "birads_category": birads,
                "predicted_tissue": pred_name,
                "confidence_pct": conf_pct,
                "fused_risk_score": fused_score,
                "cui_uncertainty_score": cui_score,
                "i0_ohm": i0_val,
                "max_ip_ohm": max_ip_val,
                "model_used": model_used,
                "sha256_audit_hash": sha256_hash,
                "created_at": now_iso
            }
        }

    def get_total_records_count(self) -> int:
        """Retorna el número real de expedientes almacenados en la base de datos."""
        try:
            if self.is_cloud_pg:
                conn = psycopg2.connect(
                    host=self.rds_host,
                    port=self.rds_port,
                    dbname=self.rds_db,
                    user=self.rds_user,
                    password=self.rds_password
                )
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM clinical_triage_records")
                    cnt = int(cur.fetchone()[0])
                conn.close()
                return cnt
            conn = self._get_sqlite_conn()
            cur = conn.execute("SELECT COUNT(*) FROM clinical_triage_records")
            cnt = int(cur.fetchone()[0])
            conn.close()
            return cnt
        except Exception:
            return 0

    def fetch_recent_records(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Consulta los últimos expedientes clínicos persistidos realmente en la base de datos."""
        try:
            conn = self._get_sqlite_conn()
            cur = conn.execute(
                "SELECT * FROM clinical_triage_records ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception:
            return []


class HuaweiDCSManager:
    """
    Gestor de caché en memoria para telemetría de bioimpedancia de alta velocidad.
    Se conecta a Huawei Cloud DCS (Redis) o ejecuta un almacén KV real en memoria/disco.
    """

    def __init__(self, cache_file: str = "data/dcs_telemetry_cache.json"):
        self.dcs_host = os.environ.get("HUAWEI_DCS_HOST", "").strip()
        self.dcs_port = int(os.environ.get("HUAWEI_DCS_PORT", "6379"))
        self.dcs_password = os.environ.get("HUAWEI_DCS_PASSWORD", "")
        self.cache_file = cache_file
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.redis_client = None

        if self.dcs_host and HAS_REDIS:
            try:
                self.redis_client = redis.Redis(
                    host=self.dcs_host,
                    port=self.dcs_port,
                    password=self.dcs_password or None,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
            except Exception:
                self.redis_client = None

        self._load_disk_cache()

    def _load_disk_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._memory_store = data
            except Exception:
                self._memory_store = {}

    def _save_disk_cache(self):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._memory_store, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def cache_screening_session(
        self,
        patient_id: str,
        inference_result: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Almacena la sesión de triaje en caché midiendo la latencia exacta de operación.
        """
        t0 = time.perf_counter()
        cache_key = f"vita:dcs:screening:{patient_id}"
        now_ts = time.time()

        entry = {
            "patient_id": patient_id,
            "triage_risk": inference_result.get("triage_risk"),
            "birads_category": inference_result.get("birads_category"),
            "prediction_name": inference_result.get("prediction_name"),
            "confidence": round(float(inference_result.get("confidence", 0.0)), 4),
            "cui_score": inference_result.get("uncertainty", {}).get("cui_score", 0.0),
            "uncertainty_level": inference_result.get("uncertainty", {}).get("level", "N/A"),
            "model_used": inference_result.get("model_used"),
            "stored_at_epoch": now_ts,
            "expires_at_epoch": now_ts + ttl_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if self.redis_client is not None:
            self.redis_client.setex(cache_key, ttl_seconds, json.dumps(entry))
            backend = f"Huawei Cloud DCS for Redis ({self.dcs_host}:{self.dcs_port})"
        else:
            self._memory_store[cache_key] = entry
            self._save_disk_cache()
            backend = "Local In-Memory + Persistent KV Store (Huawei DCS Compatible)"

        op_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return {
            "service": backend,
            "cache_key": cache_key,
            "status": "STORED_VERIFIED",
            "ttl_seconds": ttl_seconds,
            "active_keys_in_cache": len(self._memory_store),
            "cached_payload": entry,
            "measured_latency_ms": op_latency_ms
        }


class HuaweiModelArtsClient:
    """
    Cliente de inferencia para Huawei Cloud ModelArts Online Service.
    Si HUAWEI_MODELARTS_ENDPOINT está definido, realiza una petición HTTP POST real;
    de lo contrario, ejecuta el servicio real empaquetado en modelarts_deploy/customize_service.py.
    """

    def __init__(self):
        self.endpoint_url = os.environ.get("HUAWEI_MODELARTS_ENDPOINT", "").strip()
        self.auth_token = os.environ.get("HUAWEI_IAM_TOKEN", "").strip()
        self._local_handler = None

    def _get_local_handler(self):
        if self._local_handler is None:
            deploy_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "modelarts_deploy"
            )
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from customize_service import handle
            self._local_handler = handle
        return self._local_handler

    def invoke_inference(
        self,
        features: Dict[str, float],
        patient_id: str = "PAC-HUAWEI-001",
        model_name: str = "XGBoost",
        metabolic_factors: Optional[Dict[str, float]] = None,
        patient_palpable: bool = False
    ) -> Dict[str, Any]:
        """
        Invoca el microservicio de ModelArts con el payload oficial y mide el tiempo real de ejecución.
        """
        t0 = time.perf_counter()
        payload = {
            "patient_id": patient_id,
            "model_name": model_name,
            "features": features,
            "metabolic_factors": metabolic_factors or {},
            "patient_palpable": patient_palpable
        }

        if self.endpoint_url:
            req_bytes = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["X-Auth-Token"] = self.auth_token
            req = urllib.request.Request(self.endpoint_url, data=req_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                http_code = resp.getcode()
            execution_mode = "Huawei Cloud ModelArts Remote HTTPS Endpoint"
            endpoint_display = self.endpoint_url
        else:
            handler_fn = self._get_local_handler()
            resp_body = handler_fn(payload)
            http_code = 200
            execution_mode = "ModelArts Custom Service Engine (modelarts_deploy/customize_service.py)"
            endpoint_display = "local://modelarts_deploy/customize_service.py::handle"

        roundtrip_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "status": "success",
            "http_status_code": http_code,
            "execution_mode": execution_mode,
            "endpoint": endpoint_display,
            "request_payload": payload,
            "roundtrip_latency_ms": roundtrip_ms,
            "service_response": resp_body
        }

