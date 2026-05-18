"""Worker daemon tests."""

from __future__ import annotations

import hashlib
import secrets

import httpx
import pytest
from fastapi.testclient import TestClient

from infer_worker_vllm import MockVllmEngine, WorkerConfig, build_app
from infer_worker_vllm.weights import hash_weights, verify_weights


def _nonce() -> str:
    """H-01: produce a fresh customer_nonce per request."""
    return "0x" + secrets.token_hex(32)


@pytest.fixture
def config() -> WorkerConfig:
    return WorkerConfig(
        operator_id="op-test",
        operator_private_key_hex="11" * 32,
        gateway_id="gw-test",
        attestation_report_hash="aa" * 32,
    )


def test_engine_is_deterministic(config: WorkerConfig) -> None:
    e = MockVllmEngine(config.model_id)
    r1 = e.generate("hello world", seed=0)
    r2 = e.generate("hello world", seed=0)
    assert r1.text == r2.text
    assert r1.log_probs == r2.log_probs


def test_healthz(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["operator_id"] == "op-test"


def test_chat_completions_emits_signed_receipt(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 16,
                "customer_nonce": _nonce(),
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["choices"][0]["message"]["content"]
        receipt = payload["receipt"]
        assert receipt["operator_id"] == "op-test"
        assert receipt["operator_signature"]
        assert receipt["request_hash"]
        assert receipt["response_hash"]


def test_chat_rejects_unknown_model(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "nope",
                "messages": [{"role": "user", "content": "x"}],
                "customer_nonce": _nonce(),
            },
        )
        assert r.status_code == 400


def test_chat_rejects_missing_customer_nonce(config: WorkerConfig) -> None:
    """H-01: customer_nonce is mandatory; pydantic validation rejects missing."""
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 422


def test_chat_rejects_malformed_customer_nonce(config: WorkerConfig) -> None:
    """H-01: customer_nonce must be 64-hex-char (optionally 0x-prefixed)."""
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "x"}],
                "customer_nonce": "not-hex",
            },
        )
        assert r.status_code == 422


def test_chat_rejects_replayed_customer_nonce(config: WorkerConfig) -> None:
    """H-01: per-process LRU rejects duplicate customer_nonce."""
    app = build_app(config)
    nonce = _nonce()
    with TestClient(app) as client:
        body = {
            "model": config.model_id,
            "messages": [{"role": "user", "content": "x"}],
            "customer_nonce": nonce,
        }
        r1 = client.post("/v1/chat/completions", json=body)
        assert r1.status_code == 200, r1.text
        r2 = client.post("/v1/chat/completions", json=body)
        assert r2.status_code == 409


def test_chat_rejects_oversized_max_tokens(config: WorkerConfig) -> None:
    """H-04: max_tokens has a schema-enforced ceiling."""
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "x"}],
                "customer_nonce": _nonce(),
                "max_tokens": 1_000_000,
            },
        )
        assert r.status_code == 422


def test_last_heartbeat_endpoint(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.get("/internal/last_heartbeat")
        assert r.status_code == 200
        hb = r.json()
        assert hb["operator_id"] == "op-test"
        assert hb["signature"]


def test_verify_weights_matches_on_disk(tmp_path, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    weights = tmp_path / "model.bin"
    blob = b"fake-weight-blob"
    weights.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    config.model_path = str(weights)
    config.model_weight_hash = "0x" + digest
    # Should not raise: matches.
    verify_weights(config)


def test_verify_weights_rejects_mismatch(tmp_path, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"real-weights")
    config.model_path = str(weights)
    config.model_weight_hash = "0x" + ("11" * 32)  # not the real digest
    with pytest.raises(RuntimeError, match="weight hash mismatch"):
        verify_weights(config)


def test_verify_weights_refuses_placeholder_in_production(monkeypatch, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OROGEN_ENV", "production")
    monkeypatch.delenv("OROGEN_WORKER_SKIP_WEIGHT_CHECK", raising=False)
    # Default model_weight_hash is the placeholder.
    with pytest.raises(RuntimeError, match="placeholder default"):
        verify_weights(config)


def test_verify_weights_skips_when_env_flag_set(monkeypatch, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OROGEN_ENV", "production")
    monkeypatch.setenv("OROGEN_WORKER_SKIP_WEIGHT_CHECK", "1")
    verify_weights(config)  # no raise


def test_hash_weights_handles_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.bin").write_bytes(b"aaa")
    (tmp_path / "b.bin").write_bytes(b"bbb")
    h1 = hash_weights(tmp_path)
    h2 = hash_weights(tmp_path)
    assert h1 == h2  # deterministic
    assert len(h1) == 64


def test_chat_via_httpx_async(config: WorkerConfig) -> None:
    app = build_app(config)
    transport = httpx.ASGITransport(app=app)
    import anyio

    async def go() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "model": config.model_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "customer_nonce": _nonce(),
                },
            )
            assert r.status_code == 200

    anyio.run(go)
