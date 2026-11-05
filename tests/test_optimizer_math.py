"""Unit tests for AutoProfiler, KVBudgetPlanner, Asymmetry Role Pivot, and RuntimeDecisionRouter."""

from __future__ import annotations

from zeroswap.config import EngineConfig
from zeroswap.optimizer.profiler import AutoProfiler
from zeroswap.optimizer.router import RuntimeDecisionRouter


def test_standard_vs_inverted_asymmetry_strategy() -> None:
    cfg = EngineConfig()
    profiler = AutoProfiler(config=cfg, gpu_index=0)

    # Variant A: Heavy Orchestrator (qwen-32b) + Light Guardrail (lfm-1b)
    plan_a = profiler.generate_tune_plan("qwen-32b", "lfm-1b")
    assert plan_a.strategy_variant == "partial_layer_swap"
    assert plan_a.swapped_layers >= 1
    assert plan_a.resident_layers == plan_a.primary_model.num_layers - plan_a.swapped_layers
    assert plan_a.break_even_delta_tokens > 0

    # Variant B: Light Orchestrator (llama-3b) + Heavy Analyst/Guardrail (qwen-32b)
    plan_b = profiler.generate_tune_plan("llama-3b", "qwen-32b")
    assert plan_b.strategy_variant == "full_role_pivot"


def test_oom_free_dynamic_kv_degradation() -> None:
    cfg = EngineConfig()
    # Request a massive context window (131,072 tokens) on 32 GB VRAM
    cfg.context_priority.max_context_length = 131072
    profiler = AutoProfiler(config=cfg, gpu_index=0)

    plan = profiler.generate_tune_plan("qwen-32b", "llama-3b")
    assert plan.kv_cache_max_tokens > 0
    assert plan.degraded_quantization_fallback is not None


def test_three_path_runtime_decision_router() -> None:
    cfg = EngineConfig()
    profiler = AutoProfiler(config=cfg, gpu_index=0)
    plan = profiler.generate_tune_plan("qwen-32b", "lfm-1b")
    router = RuntimeDecisionRouter(plan, cfg)

    # Path A: Cold or large delta >= break_even_delta_tokens -> 0.0 ms bubble
    d_a = router.decide(total_tokens=plan.break_even_delta_tokens + 5000, cached_tokens=0)
    assert d_a.path == "path_a_pipelined_overwrite"
    assert d_a.estimated_pcie_bubble_ms == 0.0

    # Path B: High cache hit rate (micro delta) & small guardrail prompt -> CPU AVX-512 Eager
    d_b = router.decide(total_tokens=64000, cached_tokens=63800, guardrail_prompt_tokens=200)
    assert d_b.path == "path_b_cpu_direct_eager"
    assert d_b.estimated_pcie_bubble_ms == 0.0

    # Path C: Medium delta (< break_even_delta_tokens) but prompt > cpu_fallback_max_tokens -> Dynamic Chunking
    if plan.break_even_delta_tokens > 600:
        d_c = router.decide(
            total_tokens=10000,
            cached_tokens=10000 - (plan.break_even_delta_tokens // 2),
            guardrail_prompt_tokens=1024,
        )
        assert d_c.path == "path_c_dynamic_chunking"
        assert d_c.effective_chunk_layers == 1
