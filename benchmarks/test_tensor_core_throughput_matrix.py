"""Comprehensive Throughput & 5th-Gen Tensor Core Utilization Matrix Benchmark on RTX 5090 (sm_120).

Benchmarks:
1. `LiquidAI/d1-3B-w8a8` INT8 Tensor Core (`torch._int_mm`) Super-Batch Scaling:
   - Compares Sequential vs. Zero-Padding Packed Super-Batch across B = 1, 2, 4, 8, 16, 32 prompts
     (4 to 128 simultaneous safety decisions in a single forward pass with 0 output tokens).
2. `nvidia/Qwen3.8-27B-NVFP4` (64 layers, 15 real safetensors layers = 3,773.04 MB swapped):
   - Sweeps Batch x Sequence Length scenarios from 256 tokens up to 32,768 tokens:
     * (B=1, S=256), (B=1, S=1024), (B=4, S=512), (B=1, S=4096),
       (B=8, S=1024), (B=16, S=1024), (B=32, S=1024)
   - Compares all 4 Pipeline Modes:
     * `sync_stop_and_copy` (No Overlap)
     * `coarse_bulk_overlap` (Single Barrier at Layer 49)
     * `fine_grained_layer_chained` (Per-Layer C++/CUDA 256-Slot Event Pipeline)
     * `multi_stream_superbatch_interleaved` (Dual Compute Streams + Interleaved Microbatches)
   - Compares both Tensor Core Precision Modes on RTX 5090:
     * `bf16_tensor_core` (`torch.mm`, ~243 TFLOPS peak)
     * `fp8_e4m3_tensor_core` (`torch._scaled_mm`, ~541 TFLOPS peak)
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import torch

from zeroswap.engine.tensor_core_pipeline import (
    PipelineMode,
    PrecisionMode,
    TensorCorePipelineEngine,
)
from zeroswap.octop.control_plane import (
    OctopControlPlane,
    resolve_local_qwen38_nvfp4_path,
)
from zeroswap.octop.d1_decision_model import (
    LiquidD1DecisionEngine,
    resolve_local_d1_path,
)


SAMPLE_AGENT_PROMPTS = [
    "Implement an iterative segment tree with lazy propagation in C++20 for range minimum queries.",
    "Connect to internal DB 10.24.88.19 with token sk-live-9876543210abcdef1234567890 for PESEL 91020312345.",
    "Reverse engineer the packet authentication state machine on 192.168.10.42 for exploit payload analysis.",
    "Compute the singular value decomposition (SVD) of a dense matrix using Householder bidiagonalization.",
]


def run_d1_int8_batch_scaling_benchmark(d1_engine: LiquidD1DecisionEngine) -> List[Dict[str, Any]]:
    print("\n" + "=" * 100)
    print(" PART 1: LiquidAI/d1-3B-w8a8 INT8 Tensor Core Super-Batch Scaling (0 Output Tokens)")
    print("=" * 100)
    print(
        f"{'Batch (Prompts)':<16} | {'Questions':<10} | {'Packed Toks':<12} | "
        f"{'Seq ms':<9} | {'Packed ms':<10} | {'Speedup':<8} | {'Tok/s':<10} | {'Decisions/s':<12} | {'INT8 TOPS':<10}"
    )
    print("-" * 100)

    results: List[Dict[str, Any]] = []
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # Warm-up
    _ = d1_engine.evaluate_octop_superbatch_packed(SAMPLE_AGENT_PROMPTS[:2])

    for b in batch_sizes:
        prompts = [SAMPLE_AGENT_PROMPTS[i % len(SAMPLE_AGENT_PROMPTS)] for i in range(b)]

        # 1. Sequential multi-question evaluation (1 state at a time)
        t0 = time.perf_counter()
        _ = d1_engine.evaluate_octop_batch(prompts)
        seq_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Packed zero-padding super-batch evaluation (all B * 4 questions in 1 tree pass)
        _, telemetry = d1_engine.evaluate_octop_superbatch_packed(prompts)
        packed_ms = telemetry["wall_time_ms"]
        speedup = seq_ms / max(0.1, packed_ms)

        row = {
            "batch_prompts": b,
            "total_questions": int(telemetry["total_questions_evaluated"]),
            "total_packed_tokens": int(telemetry["total_packed_tokens"]),
            "sequential_wall_ms": round(seq_ms, 2),
            "packed_superbatch_wall_ms": packed_ms,
            "speedup_vs_sequential": round(speedup, 2),
            "throughput_tokens_per_sec": telemetry["throughput_tokens_per_sec"],
            "throughput_prompts_per_sec": telemetry["throughput_prompts_per_sec"],
            "throughput_decisions_per_sec": telemetry["throughput_decisions_per_sec"],
            "sustained_int8_tensor_core_tops": telemetry["sustained_int8_tensor_core_tops"],
        }
        results.append(row)
        print(
            f"{b:<16} | {row['total_questions']:<10} | {row['total_packed_tokens']:<12} | "
            f"{row['sequential_wall_ms']:<9.1f} | {row['packed_superbatch_wall_ms']:<10.1f} | "
            f"{row['speedup_vs_sequential']:<7.2f}x | {row['throughput_tokens_per_sec']:<10.1f} | "
            f"{row['throughput_decisions_per_sec']:<12.1f} | {row['sustained_int8_tensor_core_tops']:<10.2f}"
        )

    return results


def run_qwen38_tensor_core_pipeline_matrix(plane: OctopControlPlane) -> Dict[str, Any]:
    tc_engine = TensorCorePipelineEngine(
        streamer=plane.streamer,
        swapped_layer_indices=plane.swapped_layer_indices,
        total_layers=64,
        hidden_size=5120,
        intermediate_size=17408,
        device="cuda:0",
    )

    print("\n" + "=" * 116)
    print(
        f" PART 2: Qwen3.8-27B-NVFP4 (64 Layers, 15 Swapped = {tc_engine.swapped_slab_mb:.1f} MB) "
        f"Tensor Core Utilization & Throughput Matrix"
    )
    print(
        f" Hardware Peak Probed on RTX 5090: BF16 = {tc_engine.hardware_peak_tflops['bf16_tensor_core']:.2f} TFLOPS | "
        f"FP8 e4m3 = {tc_engine.hardware_peak_tflops['fp8_e4m3_tensor_core']:.2f} TFLOPS"
    )
    print("=" * 116)

    scenarios: List[Tuple[str, int, int]] = [
        ("Micro_CacheHit_B1_S256", 1, 256),
        ("Interactive_Turn_B1_S1024", 1, 1024),
        ("MultiAgent_Rollout_B4_S512", 4, 512),
        ("Standard_Prefill_B1_S4096", 1, 4096),
        ("Parallel_Batch_B8_S1024", 8, 1024),
        ("Heavy_Codebase_B16_S1024", 16, 1024),
        ("Max_SuperBatch_B32_S1024", 32, 1024),
    ]

    pipeline_modes: List[PipelineMode] = [
        "sync_stop_and_copy",
        "coarse_bulk_overlap",
        "fine_grained_layer_chained",
        "multi_stream_superbatch_interleaved",
    ]

    precision_modes: List[PrecisionMode] = [
        "bf16_tensor_core",
        "fp8_e4m3_tensor_core",
    ]

    matrix_results: Dict[str, List[Dict[str, Any]]] = {}

    for prec in precision_modes:
        print(f"\n--- Precision Mode: {prec} (Empirical Peak: {tc_engine.hardware_peak_tflops[prec]:.1f} TFLOPS) ---")
        print(
            f"{'Scenario (B x S)':<26} | {'Pipeline Mode':<34} | {'Tokens':<7} | "
            f"{'Wall ms':<8} | {'Stall ms':<8} | {'Tok/s':<9} | {'Sust. TFLOPS':<12} | {'TC Duty %':<9} | {'Peak Sat %':<10}"
        )
        print("-" * 138)
        prec_rows: List[Dict[str, Any]] = []

        for scen_name, b_size, s_len in scenarios:
            for p_mode in pipeline_modes:
                m = tc_engine.run_pipeline_benchmark(
                    batch_size=b_size,
                    seq_len=s_len,
                    pipeline_mode=p_mode,
                    precision_mode=prec,
                    num_microbatches=3,
                )
                row = {"scenario": scen_name, **asdict(m)}
                prec_rows.append(row)
                print(
                    f"{scen_name:<26} | {p_mode:<34} | {m.total_tokens:<7} | "
                    f"{m.total_wall_ms:<8.1f} | {m.pcie_exposed_stall_ms:<8.1f} | "
                    f"{m.throughput_tokens_per_sec:<9.1f} | {m.sustained_pipeline_tflops:<12.2f} | "
                    f"{m.tensor_core_duty_cycle_pct:<8.1f}% | {m.tensor_core_peak_saturation_pct:<9.1f}%"
                )
        matrix_results[prec] = prec_rows

    # Verify all Qwen3.8-27B-NVFP4 hardware invariants after the intensive matrix sweep
    kv_ok = plane.slab.verify_kv_cache_invariant(plane.kv_caches, plane.block_tables)
    ptrs_ok = all(plane.slab.verify_layer_pointers_invariant(idx) for idx in plane.swapped_layer_indices)
    bit_exact_ok = plane.verify_qwen_layers_bit_exact()

    print("\n" + "=" * 116)
    print(
        f" Post-Sweep Hardware Invariants: KV Cache Intact = {kv_ok} | "
        f"CUDA Graph Pointers Intact = {ptrs_ok} | Qwen3.8 Bit-Exact Checksum = {bit_exact_ok}"
    )
    print("=" * 116)

    return {
        "hardware_peak_tflops": tc_engine.hardware_peak_tflops,
        "swapped_slab_mb": round(tc_engine.swapped_slab_mb, 2),
        "post_sweep_invariants": {
            "kv_cache_invariant_ok": kv_ok,
            "cuda_graph_pointers_invariant_ok": ptrs_ok,
            "qwen_weights_bit_exact_restored": bit_exact_ok,
        },
        "precision_matrices": matrix_results,
    }


def main() -> None:
    d1_engine = LiquidD1DecisionEngine(
        model_id_or_path=resolve_local_d1_path(),
        device="cuda:0",
        compile_model=False,
    )
    plane = OctopControlPlane(
        d1_engine=d1_engine,
        qwen_snapshot_dir=resolve_local_qwen38_nvfp4_path(),
        num_swapped_qwen_layers=15,  # 3,773.04 MB of real Qwen3.8-27B-NVFP4 safetensors layers
        kv_cache_mb=1024,
        device="cuda:0",
    )

    d1_scaling = run_d1_int8_batch_scaling_benchmark(d1_engine)
    qwen_matrix = run_qwen38_tensor_core_pipeline_matrix(plane)

    full_report = {
        "gpu_name": plane.plan.gpu_name,
        "pcie_bandwidth_gbps": plane.plan.pcie_bandwidth_gbps,
        "d1_int8_superbatch_scaling": d1_scaling,
        "qwen38_tensor_core_matrix": qwen_matrix,
    }

    out_file = Path(__file__).resolve().parent / "tensor_core_throughput_matrix.json"
    out_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved complete Tensor Core & Throughput matrix report to: {out_file}")


if __name__ == "__main__":
    main()
