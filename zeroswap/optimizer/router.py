"""RuntimeDecisionRouter: Dynamic 3-Path Execution Heuristic based on Prefix Cache Hit Rate & Delta N.

For every agentic step, inspects the uncached token delta (`Delta N = N_total - N_cached_radix`)
against the Break-Even Delta Token Threshold (`Delta N_crit`) and selects:
- Path A (`path_a_pipelined_overwrite`): When `Delta N >= Delta N_crit`. Full asynchronous HtoD
  weight overwrite overlapped under resident layer prefill (0.00 ms PCIe bubble).
- Path B (`path_b_cpu_direct_eager`): When `Delta N < Delta N_crit` and the guardrail evaluation
  prompt is small (`<= cpu_fallback_max_tokens`). Runs the lightweight guardrail directly on
  AVX-512 CPU threads without touching GPU VRAM or the PCIe bus.
- Path C (`path_c_dynamic_chunking`): When `Delta N < Delta N_crit` and the prompt exceeds the
  CPU micro-batch limit. Slices the GPU weight swap into 1-layer micro-packets (`~15 ms` per chunk),
  lowering the effective break-even threshold by `1 / N_swapped_layers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from zeroswap.config import EngineConfig, RoutingPath, TunePlan


@dataclass
class RoutingDecision:
    path: RoutingPath
    total_tokens: int
    cached_tokens: int
    delta_tokens: int
    cache_hit_rate: float
    break_even_delta_tokens: int
    effective_chunk_layers: int
    estimated_pcie_bubble_ms: float
    reason: str


class RuntimeDecisionRouter:
    """Selects the optimal execution path per request based on Radix/Prefix Cache hit rate and Delta N."""

    def __init__(self, plan: TunePlan, config: EngineConfig | None = None) -> None:
        self.plan = plan
        self.config = config or EngineConfig()

    def decide(
        self,
        total_tokens: int,
        cached_tokens: int = 0,
        guardrail_prompt_tokens: int | None = None,
    ) -> RoutingDecision:
        total_tokens = max(1, int(total_tokens))
        cached_tokens = max(0, min(total_tokens, int(cached_tokens)))
        delta_tokens = max(1, total_tokens - cached_tokens)
        cache_hit_rate = cached_tokens / float(total_tokens)
        guardrail_tokens = guardrail_prompt_tokens if guardrail_prompt_tokens is not None else delta_tokens

        crit = max(1, self.plan.break_even_delta_tokens)
        resident_compute_ms = (
            delta_tokens / max(1.0, self.plan.prefill_throughput_tokens_per_sec)
        ) * 1000.0

        # Path A: Delta N >= Delta N_crit -> Full Pipelined Weight Overwrite (0 ms bubble)
        if delta_tokens >= crit:
            return RoutingDecision(
                path="path_a_pipelined_overwrite",
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                delta_tokens=delta_tokens,
                cache_hit_rate=round(cache_hit_rate, 4),
                break_even_delta_tokens=crit,
                effective_chunk_layers=self.plan.swapped_layers,
                estimated_pcie_bubble_ms=0.0,
                reason=(
                    f"Delta N ({delta_tokens}) >= Delta N_crit ({crit}): "
                    f"PCIe transfer ({self.plan.transfer_time_swapped_slab_ms:.1f} ms) is 100% hidden "
                    f"under resident prefill compute ({resident_compute_ms:.1f} ms)."
                ),
            )

        # Path B: Delta N < Delta N_crit and small guardrail prompt -> CPU AVX-512 Direct Eager
        if guardrail_tokens <= self.config.streaming.cpu_fallback_max_tokens:
            return RoutingDecision(
                path="path_b_cpu_direct_eager",
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                delta_tokens=delta_tokens,
                cache_hit_rate=round(cache_hit_rate, 4),
                break_even_delta_tokens=crit,
                effective_chunk_layers=0,
                estimated_pcie_bubble_ms=0.0,
                reason=(
                    f"High prefix cache hit ({cache_hit_rate * 100:.1f}%, Delta N={delta_tokens} < {crit}) "
                    f"with micro-prompt ({guardrail_tokens} tokens <= {self.config.streaming.cpu_fallback_max_tokens}): "
                    f"routing guardrail to AVX-512 CPU Eager to avoid PCIe bus bubble."
                ),
            )

        # Path C: Delta N < Delta N_crit and medium prompt -> Dynamic 1-Layer Micro-Chunking
        chunk_layers = max(1, self.config.streaming.dynamic_chunk_layers)
        first_chunk_transfer_ms = self.plan.transfer_time_single_layer_ms * chunk_layers
        bubble_ms = max(0.0, first_chunk_transfer_ms - resident_compute_ms)
        return RoutingDecision(
            path="path_c_dynamic_chunking",
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            delta_tokens=delta_tokens,
            cache_hit_rate=round(cache_hit_rate, 4),
            break_even_delta_tokens=max(1, crit // max(1, self.plan.swapped_layers)),
            effective_chunk_layers=chunk_layers,
            estimated_pcie_bubble_ms=round(bubble_ms, 2),
            reason=(
                f"Delta N ({delta_tokens}) < full-slab Delta N_crit ({crit}): "
                f"slicing weight swap into {chunk_layers}-layer micro-chunks "
                f"(first chunk ready in {first_chunk_transfer_ms:.1f} ms, bubble={bubble_ms:.1f} ms)."
            ),
        )
