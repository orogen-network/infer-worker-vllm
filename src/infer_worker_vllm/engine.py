"""Mock vLLM engine.

Real implementation lives behind `vllm.LLM(...)`; this mock keeps the project
dependency-free so the daemon compiles, tests run on a laptop, and the harness
can pin behaviour exactly.

Determinism: given the same `(prompt, model_id, seed)` the engine returns the same
text and log-probs. This is what the validator-replay path expects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(slots=True)
class InferenceResult:
    text: str
    tokens: list[str]
    log_probs: list[float]
    prompt_tokens: int
    completion_tokens: int


class MockVllmEngine:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, prompt: str, *, max_tokens: int = 32, seed: int = 0) -> InferenceResult:
        # Deterministic pseudo-tokens hashed off the prompt+model.
        key = f"{self.model_id}::{prompt}::{seed}".encode()
        digest = hashlib.sha256(key).digest()
        n_tokens = min(max(4, len(prompt) // 4), max_tokens)
        tokens = [f"t{digest[i % len(digest)]:02x}" for i in range(n_tokens)]
        # Log-prob sample is sha256-bytes mapped into [-5.0, 0.0).
        log_probs = [-(b / 51.0) for b in digest[:64]]
        text = " ".join(tokens)
        return InferenceResult(
            text=text,
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=n_tokens,
        )
