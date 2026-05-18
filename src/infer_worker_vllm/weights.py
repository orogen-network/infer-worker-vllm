"""Weight-hash verification helper.

C-05 fix: workers must verify that on-disk weights actually hash to the value
declared in `WorkerConfig.model_weight_hash`, otherwise a malicious operator
who swaps weights and advertises the original hash produces receipts that
match the validator's expected hash, bypassing receipt-level mismatch
detection.

Behaviour:
- Hash function: streaming SHA-256 over the on-disk weight file (or every
  regular file in a directory, sorted by relative path) and concatenated.
- The declared `model_weight_hash` is normalised by stripping a leading "0x".
- If the placeholder default (`"ab" * 32`) is in effect, refusal is gated on
  `OROGEN_ENV=production`; otherwise it logs a warning and continues.
- If `model_path` is unset or does not exist, this is treated as a Mock
  engine — log a warning and skip (the mock has no on-disk weights).
- `OROGEN_WORKER_SKIP_WEIGHT_CHECK=1` short-circuits the check entirely
  (used by unit tests that don't ship a real weight tree).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Protocol


class _WeightsConfig(Protocol):
    model_weight_hash: str
    model_path: str | None


PLACEHOLDER_HASH = "ab" * 32
_LOG = logging.getLogger("infer_worker_vllm.weights")
_CHUNK = 1 << 20  # 1 MiB


def _normalize(declared: str) -> str:
    s = declared.lower()
    return s[2:] if s.startswith("0x") else s


def hash_weights(path: Path) -> str:
    """Compute SHA-256 of `path`.

    For a regular file, hash the file bytes.
    For a directory, hash every regular file inside it, sorted by relative
    path; each file is mixed into a single digest as
    `len(rel_path) || rel_path || len(bytes) || file_bytes`.
    """
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        files = sorted(
            (p for p in path.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        for p in files:
            rel = p.relative_to(path).as_posix().encode("utf-8")
            h.update(len(rel).to_bytes(8, "big"))
            h.update(rel)
            size = p.stat().st_size
            h.update(size.to_bytes(8, "big"))
            with p.open("rb") as f:
                while chunk := f.read(_CHUNK):
                    h.update(chunk)
        return h.hexdigest()
    raise FileNotFoundError(f"weights path is neither file nor dir: {path}")


def verify_weights(config: _WeightsConfig) -> None:
    """Verify on-disk weights match `config.model_weight_hash`.

    Raises:
        RuntimeError: when the placeholder hash is in effect under
            `OROGEN_ENV=production`, or when the computed hash disagrees
            with the declared one.
    """
    if os.environ.get("OROGEN_WORKER_SKIP_WEIGHT_CHECK") == "1":
        _LOG.info("weight verification skipped via OROGEN_WORKER_SKIP_WEIGHT_CHECK=1")
        return

    declared = _normalize(config.model_weight_hash)
    is_placeholder = declared == PLACEHOLDER_HASH
    env = os.environ.get("OROGEN_ENV", "dev").lower()

    if is_placeholder:
        if env == "production":
            raise RuntimeError(
                "refusing to start: model_weight_hash is the placeholder default "
                f"(0x{PLACEHOLDER_HASH}) under OROGEN_ENV=production. "
                "Set a real weight hash or run with OROGEN_WORKER_SKIP_WEIGHT_CHECK=1."
            )
        _LOG.warning(
            "model_weight_hash is the placeholder default (0x%s); skipping verification (env=%s)",
            PLACEHOLDER_HASH,
            env,
        )
        return

    model_path = getattr(config, "model_path", None)
    if not model_path:
        _LOG.warning(
            "model_path not configured; skipping weight verification "
            "(Mock engines have no on-disk weights to hash)"
        )
        return
    p = Path(model_path)
    if not p.exists():
        _LOG.warning(
            "model_path=%s does not exist; skipping weight verification "
            "(Mock engines have no on-disk weights to hash)",
            model_path,
        )
        return

    computed = hash_weights(p)
    if computed != declared:
        raise RuntimeError(
            "weight hash mismatch: declared model_weight_hash=0x"
            f"{declared} but on-disk hash of {model_path} is 0x{computed}. "
            "Refusing to start to prevent operator from signing receipts "
            "against weights that do not match the advertised hash."
        )
    _LOG.info("weight verification ok: model_path=%s sha256=0x%s", model_path, computed)
