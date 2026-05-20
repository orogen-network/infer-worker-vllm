"""Heartbeat sender + simple watchdog.

The sender is an asyncio task. In production it pushes to a gateway WebSocket;
here it POSTs JSON to whatever URL the harness configures, on a tick.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from mining_types import (
    AttestationFreshness,
    Capability,
    LoadSnapshot,
    OffChainHeartbeat,
    Quantization,
    WatchdogState,
)

from infer_worker_vllm.config import WorkerConfig


def _gateway_auth_headers(config: WorkerConfig) -> dict[str, str]:
    token = (
        config.gateway_auth_token
        or os.environ.get("GATEWAY_INTERNAL_AUTH_TOKEN", "")
        or os.environ.get("INTERNAL_AUTH_TOKEN", "")
    ).strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def build_heartbeat(config: WorkerConfig, load: LoadSnapshot) -> OffChainHeartbeat:
    now_ms = int(time.time() * 1000)
    hb = OffChainHeartbeat(
        operator_id=config.operator_id,
        capabilities=[
            Capability(
                base_model_id=config.model_id,
                quantization=Quantization.FP16,
                max_context_tokens=8192,
                max_concurrent_requests=8,
                deterministic_mode=True,
            )
        ],
        current_load=load,
        kv_cache_pressure=load.gpu_utilization_pct / 100.0,
        attestation_freshness=AttestationFreshness(
            last_attested_at_ms=now_ms,
            expires_at_ms=now_ms + 7 * 86400 * 1000,
            current_report_hash=config.attestation_report_hash,
        ),
        watchdog_state=WatchdogState(vllm_pid_alive=True, vllm_last_log_ms=now_ms),
        endpoint_url=config.base_url,
        price_per_million_tokens=2_000_000,
        geo_region="US",
    )
    return hb.sign(config.operator_private_key_hex)


class HeartbeatPusher:
    """Background asyncio task that POSTs heartbeats to the gateway."""

    def __init__(
        self,
        config: WorkerConfig,
        gateway_url: str,
        load_provider: Any,
        interval_s: float | None = None,
    ) -> None:
        self.config = config
        self.gateway_url = gateway_url
        self.load_provider = load_provider
        self.interval_s = interval_s or config.heartbeat_interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.last_hb: OffChainHeartbeat | None = None

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=2.0) as client:
            while not self._stop.is_set():
                hb = build_heartbeat(self.config, self.load_provider())
                self.last_hb = hb
                try:
                    await client.post(
                        f"{self.gateway_url}/internal/heartbeat",
                        json=hb.model_dump(mode="json"),
                        headers=_gateway_auth_headers(self.config),
                    )
                except httpx.HTTPError:
                    # gateway may be slow / not up yet; we just retry next tick.
                    pass
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                except TimeoutError:
                    pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except TimeoutError:
                self._task.cancel()
