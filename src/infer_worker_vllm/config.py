"""Worker configuration. No env-var parsing here; injected by the harness/tests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkerConfig:
    operator_id: str
    operator_private_key_hex: str
    gateway_id: str
    attestation_report_hash: str
    model_id: str = "mock-model-7b"
    model_weight_hash: str = "0x" + "ab" * 32
    kernel_pack_hash: str = "0x" + "cd" * 32
    heartbeat_interval_s: float = 12.0
    watchdog_interval_s: float = 5.0
    base_url: str = ""  # filled by uvicorn host after start
    capabilities: list[str] = field(default_factory=lambda: ["mock-model-7b"])
    # On-disk path to the weights file or HF-style directory. If unset, the
    # weight-hash verification step at startup is skipped (Mock engines have
    # no weights to verify). See `weights.verify_weights`.
    model_path: str | None = None
