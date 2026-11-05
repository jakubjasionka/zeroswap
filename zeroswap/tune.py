"""CLI Auto-Tuner: `python -m zeroswap.tune --primary qwen-32b --secondary lfm-1b --gpu 0`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from zeroswap.config import EngineConfig
from zeroswap.optimizer.profiler import AutoProfiler
from zeroswap.optimizer.router import RuntimeDecisionRouter


def format_tune_report(plan, router: RuntimeDecisionRouter) -> str:
    strategy_label = (
        f"Swap {plan.swapped_layers} layers ({plan.dynamic_slab_mb / 1024.0:.2f} GB buffer)"
        if plan.strategy_variant == "partial_layer_swap"
        else f"Full Role Pivot (Secondary resident, {plan.swapped_layers} layers dynamic)"
    )
    res_end = max(0, plan.resident_layers - 1)
    lines = [
        "[ZeroSwap] Profiling complete:",
        f"  - GPU Device: {plan.gpu_name} ({plan.total_vram_mb / 1024.0:.1f} GB VRAM)",
        f"  - PCIe Bandwidth: {plan.pcie_bandwidth_gbps:.1f} GB/s (Detected {plan.pcie_generation_estimate})",
        f"  - Primary Model ({plan.primary_model.model_id}): "
        f"{plan.primary_model.num_layers} layers, ~{plan.primary_model.layer_size_mb:.0f} MB/layer "
        f"({plan.primary_model.total_weight_mb / 1024.0:.2f} GB total)",
        f"  - Guardrail Model ({plan.secondary_model.model_id}): "
        f"{plan.secondary_model.total_weight_mb / 1024.0:.2f} GB total footprint "
        f"({plan.secondary_model.num_layers} layers)",
        f"  - Asymmetry Variant: {plan.strategy_variant}",
        f"  - Optimal Strategy: {strategy_label}",
        f"  - KV Cache Headroom: {plan.kv_cache_max_tokens:,} tokens preserved "
        f"({plan.kv_cache_blocks:,} PagedAttention blocks, {plan.kv_cache_pool_mb / 1024.0:.2f} GB pool)",
        f"  - Break-Even Token Delta (Delta N_crit): {plan.break_even_delta_tokens:,} tokens "
        f"(Prefill: {plan.prefill_throughput_tokens_per_sec:,.0f} tok/s, "
        f"H2D Slab Transfer: {plan.transfer_time_swapped_slab_ms:.1f} ms)",
        f"  - Latency impact: {plan.estimated_net_ttft_overhead_ms:.2f} ms "
        f"(Fully overlapped during layer 0-{res_end} forward)",
    ]
    if plan.degraded_quantization_fallback:
        lines.append(f"  - OOM-Free Fallback Active: {plan.degraded_quantization_fallback}")

    # Sample runtime routing decisions across cache hit scenarios
    d_cold = router.decide(total_tokens=20000, cached_tokens=0)
    d_micro = router.decide(total_tokens=64000, cached_tokens=63750, guardrail_prompt_tokens=250)
    d_med = router.decide(
        total_tokens=32000,
        cached_tokens=32000 - max(200, plan.break_even_delta_tokens // 2),
        guardrail_prompt_tokens=1024,
    )
    lines.extend(
        [
            "",
            "[ZeroSwap] Dynamic Runtime Routing Heuristics:",
            f"  - Cold/Deep Extension (Delta N=20,000): {d_cold.path} "
            f"(PCIe bubble: {d_cold.estimated_pcie_bubble_ms:.2f} ms)",
            f"  - High Cache Hit Micro-Step (99.6% hit, Delta N=250): {d_micro.path} "
            f"(PCIe bubble: {d_micro.estimated_pcie_bubble_ms:.2f} ms)",
            f"  - Medium Cache Hit Step (Delta N={d_med.delta_tokens}): {d_med.path} "
            f"(Chunk={d_med.effective_chunk_layers} layer, bubble: {d_med.estimated_pcie_bubble_ms:.2f} ms)",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zeroswap.tune",
        description="Zero-Config Auto-Tuning Profiler for ZeroSwap + vLLM",
    )
    parser.add_argument(
        "--primary",
        type=str,
        default="qwen-32b",
        help="Primary orchestrator model ID or path to HF directory/config.json",
    )
    parser.add_argument(
        "--secondary",
        type=str,
        default="lfm-1b",
        help="Secondary guardrail/decision model ID or path to HF directory/config.json",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA GPU device index (default: 0)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to engine YAML config (default: configs/default_engine.yaml)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Override target max_context_length for KV cache reservation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plan.json",
        help="Output path for generated optimization plan JSON (default: plan.json)",
    )
    args = parser.parse_args(argv)

    if args.config and Path(args.config).is_file():
        cfg = EngineConfig.from_yaml(args.config)
    else:
        default_yaml = Path(__file__).resolve().parent.parent / "configs" / "default_engine.yaml"
        cfg = EngineConfig.from_yaml(default_yaml) if default_yaml.is_file() else EngineConfig()

    if args.max_context is not None:
        cfg.context_priority.max_context_length = args.max_context

    profiler = AutoProfiler(config=cfg, gpu_index=args.gpu)
    plan = profiler.generate_tune_plan(args.primary, args.secondary)
    plan.to_json(args.output)

    router = RuntimeDecisionRouter(plan, cfg)
    print(format_tune_report(plan, router))
    print(f"\n[ZeroSwap] Saved optimization plan to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
