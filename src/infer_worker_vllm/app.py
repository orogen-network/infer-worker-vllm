"""Worker FastAPI app.

Exposes:
- `POST /v1/chat/completions` — OpenAI-compatible (mocked).
- `GET /healthz` — liveness.
- `GET /internal/last_heartbeat` — for harness inspection.
- `POST /internal/register` — manual register; harness fixtures call this.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from mining_types import KvMetadata, LoadSnapshot, Receipt
from mining_types.crypto import sha256_hex
from pydantic import BaseModel, Field

from infer_worker_vllm.config import WorkerConfig
from infer_worker_vllm.engine import MockVllmEngine
from infer_worker_vllm.heartbeat import HeartbeatPusher, build_heartbeat
from infer_worker_vllm.weights import verify_weights


# H-01: bounded LRU of recently-seen customer_nonces, sized so a busy worker
# can absorb several minutes of traffic without false-rejects.
_NONCE_LRU_MAX = 10_000


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # H-03/H-04: schema bounds — worker MUST refuse oversized prompts and
    # token budgets, and MUST require a well-formed customer_nonce. Without
    # these caps the worker will sign a receipt for any input.
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=8192)
    customer_nonce: str = Field(pattern=r"^0x?[a-fA-F0-9]{64}$")
    seed: int = Field(default=0, ge=0, le=2**63 - 1)


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]
    receipt: dict[str, Any]


def build_app(config: WorkerConfig, gateway_url: str | None = None) -> FastAPI:
    # C-05: verify on-disk weights match config.model_weight_hash before
    # accepting any requests. Mock engines + dev-time placeholder hashes
    # are tolerated (see weights.verify_weights for the exact rules).
    verify_weights(config)
    engine = MockVllmEngine(model_id=config.model_id)
    state: dict[str, Any] = {
        "requests_total": 0,
        "active_requests": 0,
        "last_receipt": None,
    }
    # H-01: per-process replay-window cache. OrderedDict gives O(1) move-to-end
    # so the LRU eviction is cheap.
    nonce_cache: OrderedDict[str, int] = OrderedDict()
    pusher: HeartbeatPusher | None = None

    def _load() -> LoadSnapshot:
        return LoadSnapshot(
            active_requests=state["active_requests"],
            queue_depth=0,
            p50_ttft_ms=10,
            p99_ttft_ms=50,
            p50_itl_ms=5,
            p99_itl_ms=20,
            gpu_memory_used_gb=12.0,
            gpu_utilization_pct=42.0,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        nonlocal pusher
        if gateway_url:
            pusher = HeartbeatPusher(config, gateway_url, _load)
            pusher.start()
        try:
            yield
        finally:
            if pusher is not None:
                await pusher.stop()

    app = FastAPI(title="infer-worker-vllm", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "operator_id": config.operator_id, "model": config.model_id}

    @app.get("/internal/last_heartbeat")
    async def last_heartbeat() -> dict[str, Any]:
        if pusher is None or pusher.last_hb is None:
            return build_heartbeat(config, _load()).model_dump(mode="json")
        return pusher.last_hb.model_dump(mode="json")

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def chat_completions(req: ChatRequest) -> ChatResponse:
        if req.model != config.model_id:
            raise HTTPException(
                status_code=400, detail=f"unsupported model {req.model!r}",
            )
        # H-01: reject replayed customer_nonce within the LRU window. The
        # Pydantic Field pattern guarantees the value is well-formed.
        customer_nonce = req.customer_nonce
        if customer_nonce in nonce_cache:
            raise HTTPException(
                status_code=409, detail="customer_nonce already used",
            )
        nonce_cache[customer_nonce] = int(time.time())
        while len(nonce_cache) > _NONCE_LRU_MAX:
            nonce_cache.popitem(last=False)
        # H-04: cap prompt length defensively (the schema bound on each
        # message is `str` with implicit pydantic limits; the worker MUST
        # also bound the concatenated prompt).
        prompt = "\n".join(m.content for m in req.messages)
        if len(prompt) > 1_000_000:
            raise HTTPException(status_code=400, detail="prompt too large")
        state["requests_total"] += 1
        state["active_requests"] += 1
        try:
            result = engine.generate(prompt, max_tokens=req.max_tokens, seed=req.seed)

            req_bytes = (prompt + f"|{req.model}|{req.seed}").encode("utf-8")
            request_hash = sha256_hex(req_bytes)
            response_hash = sha256_hex(result.text.encode("utf-8"))

            # H-02: job_id is unpredictable (not bound to wall-clock).
            job_id = secrets.token_hex(32)

            receipt = Receipt(
                job_id=job_id,
                operator_id=config.operator_id,
                model_id=config.model_id,
                model_weight_hash=config.model_weight_hash,
                customer_nonce=customer_nonce,
                request_hash=request_hash,
                response_hash=response_hash,
                log_probs_sample=result.log_probs,
                kv_metadata=KvMetadata(cache_hit=False, kv_blocks_used=4),
                kernel_pack_hash=config.kernel_pack_hash,
                gpu_model="mock-H100",
                driver_version="550.54.15",
                cuda_version="12.4",
                attestation_report_hash=config.attestation_report_hash,
                timestamp_ms=int(time.time() * 1000),
                gateway_id=config.gateway_id,
            ).sign(config.operator_private_key_hex)
            state["last_receipt"] = receipt

            return ChatResponse(
                id=job_id,
                model=req.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                receipt=receipt.model_dump(mode="json"),
            )
        finally:
            state["active_requests"] -= 1

    return app


# Helper for type checkers that want a typed lifespan
LifespanHandler = Callable[[FastAPI], Awaitable[None]]
