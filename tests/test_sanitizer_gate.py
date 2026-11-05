"""Unit tests for DataSanitizationGate, `@gate.intercept`, Self-Correction Loop, and vLLM Worker Extension."""

from __future__ import annotations

from typing import Any, Dict
import pytest
import torch
import torch.nn as nn

from zeroswap.guardrail.gate import GuardrailGate
from zeroswap.guardrail.sanitizer import DataSanitizationGate
from zeroswap.vllm_integration.worker_ext import ZeroSwapWorkerExtension


def test_pii_and_credential_masking_and_unmasking() -> None:
    sanitizer = DataSanitizationGate()
    raw = (
        "Use key sk-abcdefghijklmnopqrstuvwxyz123456 on server 192.168.10.42 "
        "and notify admin@corp.internal or PESEL 89010234567."
    )
    res = sanitizer.sanitize(raw)
    assert res.redacted_count == 4
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in res.sanitized_text
    assert "192.168.10.42" not in res.sanitized_text
    assert "admin@corp.internal" not in res.sanitized_text
    assert "89010234567" not in res.sanitized_text

    # Frontier response uses the synthetic tokens
    frontier_reply = f"Connected to {res.sanitized_text}"
    restored = sanitizer.unmask(frontier_reply, res.mock_to_real)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" in restored
    assert "192.168.10.42" in restored
    assert "admin@corp.internal" in restored


def test_anti_censorship_microtask_decomposition() -> None:
    sanitizer = DataSanitizationGate()
    query = (
        "Analyze the buffer vulnerability in the parser. "
        "Reverse engineer the payload framing. "
        "Check if an exploit can bypass validation."
    )
    tasks = sanitizer.decompose_into_abstract_microtasks(query, max_tasks=3)
    assert len(tasks) == 3
    joined = " ".join(tasks).lower()
    assert "vulnerability" not in joined
    assert "exploit" not in joined
    assert "state_transition_anomaly" in joined


import asyncio


def test_gate_intercept_self_correction_loop() -> None:
    local_gate = GuardrailGate()

    @local_gate.intercept(safety_model="lfm-1b-guard", mode="pipelined", max_retries=2)
    async def agent_step(prompt: str, kv_context: Dict[str, Any]) -> str:
        return f"OK: {prompt}"

    ctx: Dict[str, Any] = {"total_tokens": 4096, "cached_tokens": 3900}
    out = asyncio.run(agent_step("Run dangerous command: rm -rf / on host 10.0.0.12", ctx))
    assert "OK:" in out
    assert "10.0.0.12" in out  # Unmasked after execution
    assert len(ctx["self_correction_history"]) == 1
    assert ctx["self_correction_history"][0]["reason_code"] == "DESTRUCTIVE_SHELL"


def test_vllm_worker_extension_lifecycle() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class DummyVLLMModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(8)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    class DummyVLLMWorker(ZeroSwapWorkerExtension):
        def __init__(self) -> None:
            self.model = DummyVLLMModel().to(device)
            self.kv_cache = [torch.zeros((16, 16, 4, 16), device=device)]

        def get_model(self) -> nn.Module:
            return self.model

    worker = DummyVLLMWorker()
    init_info = worker.zeroswap_init(num_swapped_layers=2, pin_memory=(device.type == "cuda"))
    assert init_info["status"] == "initialized"
    assert init_info["swapped_layer_indices"] == [6, 7]

    step_res = worker.zeroswap_run_guardrail_step(
        "Check API key sk-1234567890abcdef12345678 on 10.1.2.3",
        total_tokens=4096,
        cached_tokens=1024,
    )
    assert step_res["approved"] is True
    assert step_res["redacted_entities"] == 2
    assert step_res["kv_and_cudagraph_invariant_preserved"] is True

    # Run model forward pass to trigger LayerInterceptor restoration
    x = torch.randn((2, 64), device=device)
    with torch.no_grad():
        _ = worker.model(x)
    inv = worker.zeroswap_verify_invariants()
    assert inv["invariant_preserved"] is True
