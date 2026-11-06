"""Rigorous tests for TensorCorePipelineEngine & Superbatch Packed D1 Tensor Core Utilization."""

from __future__ import annotations

import pytest
import torch

from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.static_slab import StaticSlab
from zeroswap.engine.tensor_core_pipeline import TensorCoreMetrics, TensorCorePipelineEngine
from zeroswap.octop.d1_decision_model import LiquidD1DecisionEngine, resolve_local_d1_path


class _DummySwapLayer(torch.nn.Module):
    def __init__(self, hidden_size: int = 5120) -> None:
        super().__init__()
        # ~30 MB per dummy layer for fast test execution while exercising real PCIe DMA + Tensor Cores
        self.weight_a = torch.nn.Parameter(
            torch.randint(0, 255, (hidden_size, 3072), dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_b = torch.nn.Parameter(
            torch.randint(0, 255, (hidden_size, 3072), dtype=torch.uint8),
            requires_grad=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_tensor_core_pipeline_modes_bf16_and_fp8() -> None:
    """Verifies all 4 Tensor Core pipeline modes across BF16 and FP8 e4m3 Tensor Cores on RTX 5090."""
    device = torch.device("cuda:0")
    swapped_indices = [12, 13, 14, 15]
    total_layers = 16

    layers = torch.nn.ModuleList([_DummySwapLayer().to(device) for _ in range(total_layers)])
    slab = StaticSlab(device=device)
    pinned = PinnedHostPool()
    for idx in swapped_indices:
        slab.register_swappable_layer(idx, layers[idx])
        pinned.mirror_primary_layer(idx, layers[idx])

    streamer = PCIeStreamer(slab=slab, host_pool=pinned, device=device)
    engine = TensorCorePipelineEngine(
        streamer=streamer,
        swapped_layer_indices=swapped_indices,
        total_layers=total_layers,
        hidden_size=5120,
        intermediate_size=17408,
        device=device,
    )

    assert engine.hardware_peak_tflops["bf16_tensor_core"] > 100.0
    assert engine.hardware_peak_tflops["fp8_e4m3_tensor_core"] > 200.0
    assert engine.hardware_peak_tflops["fp8_e4m3_tensor_core"] > engine.hardware_peak_tflops["bf16_tensor_core"]

    for precision in ("bf16_tensor_core", "fp8_e4m3_tensor_core"):
        results: dict[str, TensorCoreMetrics] = {}
        for mode in (
            "sync_stop_and_copy",
            "coarse_bulk_overlap",
            "fine_grained_layer_chained",
            "multi_stream_superbatch_interleaved",
        ):
            m = engine.run_pipeline_benchmark(
                batch_size=4,
                seq_len=512,
                pipeline_mode=mode,  # type: ignore[arg-type]
                precision_mode=precision,  # type: ignore[arg-type]
                num_microbatches=3,
            )
            results[mode] = m
            assert m.total_tokens >= 2048
            assert m.throughput_tokens_per_sec > 0.0
            assert m.sustained_pipeline_tflops > 0.0
            assert 0.0 < m.tensor_core_duty_cycle_pct <= 100.0

        # Pipelined modes must achieve strictly higher Tensor Core duty cycle and higher sustained TFLOPS
        # than synchronous stop-and-copy
        sync_m = results["sync_stop_and_copy"]
        chained_m = results["fine_grained_layer_chained"]
        super_m = results["multi_stream_superbatch_interleaved"]

        assert chained_m.tensor_core_duty_cycle_pct > sync_m.tensor_core_duty_cycle_pct
        assert super_m.tensor_core_duty_cycle_pct > sync_m.tensor_core_duty_cycle_pct
        assert super_m.sustained_pipeline_tflops > sync_m.sustained_pipeline_tflops
        assert super_m.throughput_tokens_per_sec > sync_m.throughput_tokens_per_sec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_d1_superbatch_packed_tensor_core_acceleration() -> None:
    """Verifies that `evaluate_octop_superbatch_packed` matches `evaluate_octop_batch` decisions while accelerating throughput."""
    d1 = LiquidD1DecisionEngine(
        model_id_or_path=resolve_local_d1_path(),
        device="cuda:0",
        compile_model=False,
    )
    prompts = [
        "Write a Python function to compute Fibonacci numbers with memoization.",
        "Customer Jan Kowalski (PESEL 89041209876, IBAN PL61109010140000071219812874) filed court claim.",
        "Explain how CUDA Event ring buffers synchronize HtoD DMA streams with Tensor Core GEMMs.",
        "Internal seedinfer.com trading strategy weights and live API key sk-live-9876543210fedcba9876543210.",
        "Convert a SQLite query into an async SQLAlchemy 2.0 select statement.",
        "Generate a PyTorch custom C++ extension setup.py targeting sm_120 Blackwell.",
    ]

    # Warmup
    _ = d1.evaluate_octop_batch(prompts[:2])
    _ = d1.evaluate_octop_superbatch_packed(prompts[:2])

    seq_decisions = d1.evaluate_octop_batch(prompts)
    packed_decisions, packed_stats = d1.evaluate_octop_superbatch_packed(prompts)

    assert len(seq_decisions) == len(packed_decisions) == len(prompts)
    for s_dec, p_dec in zip(seq_decisions, packed_decisions):
        assert s_dec.approved_for_external_api == p_dec.approved_for_external_api
        assert s_dec.decision_action == p_dec.decision_action

    assert packed_stats["total_questions_evaluated"] == len(prompts) * 4
    assert packed_stats["sustained_int8_tensor_core_tops"] > 0.0
    assert packed_stats["throughput_prompts_per_sec"] > 0.0
