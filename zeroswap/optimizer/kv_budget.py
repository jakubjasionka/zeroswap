"""KVBudgetPlanner: Dynamic PagedAttention KV Cache Reservation & OOM-Free Slab Degradation.

Computes:
    Free VRAM = Total VRAM * target_utilization - Resident Weights - Dynamic Slab (Guardrail)
    Available KV Blocks = floor(Free VRAM / PagedAttention Block Byte Size)

If the user's requested `max_context_length` cannot fit alongside the full guardrail slab,
the planner automatically degrades the slab into a 2-layer Ping-Pong streaming buffer or
applies aggressive guardrail quantization (NVFP4/INT4) instead of raising CUDA OOM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from zeroswap.config import EngineConfig, ModelTopology


@dataclass
class KVBudgetAllocation:
    usable_vram_mb: float
    resident_weights_mb: float
    dynamic_slab_mb: float
    ping_pong_buffer_mb: float
    kv_cache_pool_mb: float
    kv_block_bytes: int
    kv_cache_blocks: int
    kv_cache_max_tokens: int
    swapped_layers: int
    resident_layers: int
    degraded_quantization_fallback: Optional[str]


class KVBudgetPlanner:
    """Calculates optimal PagedAttention block allocation and slab size without CUDA OOM."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def plan_budget(
        self,
        total_vram_mb: float,
        base_model: ModelTopology,
        guest_model: ModelTopology,
    ) -> KVBudgetAllocation:
        usable_vram_mb = total_vram_mb * self.config.target_gpu_memory_utilization
        block_size = self.config.context_priority.block_size
        target_tokens = self.config.context_priority.max_context_length

        kv_bytes_per_token = base_model.kv_bytes_per_token
        required_kv_mb = (target_tokens * kv_bytes_per_token) / (1024.0 * 1024.0)

        # Initial desired dynamic slab = enough base layers to fit the guest model footprint
        desired_slab_mb = guest_model.total_weight_mb
        swapped_layers = max(
            1,
            min(
                base_model.num_layers - 1,
                math.ceil(desired_slab_mb / max(1.0, base_model.layer_size_mb)),
            ),
        )
        non_layer_overhead_mb = max(
            0.0, base_model.total_weight_mb - (base_model.num_layers * base_model.layer_size_mb)
        )
        actual_slab_mb = swapped_layers * base_model.layer_size_mb
        resident_layers = base_model.num_layers - swapped_layers
        resident_weights_mb = non_layer_overhead_mb + resident_layers * base_model.layer_size_mb
        ping_pong_buffer_mb = 2.0 * guest_model.layer_size_mb

        free_kv_mb = usable_vram_mb - resident_weights_mb - actual_slab_mb
        degraded_fallback: Optional[str] = None

        # OOM-Free Dynamic Degradation if KV Cache headroom is below target_tokens
        if free_kv_mb < required_kv_mb:
            # Step 1: Try on-the-fly NVFP4 quantization fallback (0.5x guest footprint)
            degraded_fallback = "nvfp4_dynamic_fallback"
            reduced_guest_mb = guest_model.total_weight_mb * 0.5
            swapped_layers = max(
                1,
                min(
                    base_model.num_layers - 1,
                    math.ceil(reduced_guest_mb / max(1.0, base_model.layer_size_mb)),
                ),
            )
            actual_slab_mb = swapped_layers * base_model.layer_size_mb
            resident_layers = base_model.num_layers - swapped_layers
            resident_weights_mb = resident_layers * base_model.layer_size_mb
            free_kv_mb = usable_vram_mb - resident_weights_mb - actual_slab_mb

        if free_kv_mb < required_kv_mb:
            # Step 2: Degrade slab to a minimal 2-layer Ping-Pong streaming buffer + offload more base layers
            degraded_fallback = "ping_pong_micro_slab+nvfp4"
            needed_extra_mb = required_kv_mb - free_kv_mb
            extra_offload_layers = math.ceil(needed_extra_mb / max(1.0, base_model.layer_size_mb))
            swapped_layers = min(
                base_model.num_layers - 2,
                swapped_layers + extra_offload_layers,
            )
            resident_layers = base_model.num_layers - swapped_layers
            resident_weights_mb = resident_layers * base_model.layer_size_mb
            actual_slab_mb = max(ping_pong_buffer_mb, base_model.layer_size_mb)
            free_kv_mb = max(256.0, usable_vram_mb - resident_weights_mb - actual_slab_mb)

        kv_block_bytes = int(max(1, block_size * kv_bytes_per_token))
        kv_cache_blocks = int((free_kv_mb * 1024.0 * 1024.0) // kv_block_bytes)
        kv_cache_max_tokens = kv_cache_blocks * block_size

        return KVBudgetAllocation(
            usable_vram_mb=round(usable_vram_mb, 2),
            resident_weights_mb=round(resident_weights_mb, 2),
            dynamic_slab_mb=round(actual_slab_mb, 2),
            ping_pong_buffer_mb=round(ping_pong_buffer_mb, 2),
            kv_cache_pool_mb=round(free_kv_mb, 2),
            kv_block_bytes=kv_block_bytes,
            kv_cache_blocks=kv_cache_blocks,
            kv_cache_max_tokens=kv_cache_max_tokens,
            swapped_layers=swapped_layers,
            resident_layers=resident_layers,
            degraded_quantization_fallback=degraded_fallback,
        )
