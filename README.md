# infer-worker-vllm

Operator daemon that wraps a (mocked) vLLM inference engine, exposes an OpenAI-compatible
HTTP endpoint, signs each response as an RFC-0001 receipt, and emits RFC-0003 heartbeats
upstream.

For tests the real vLLM engine is replaced with a deterministic mock so this package
has no GPU / heavy ML dependency.
