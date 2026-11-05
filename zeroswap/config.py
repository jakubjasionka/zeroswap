"""Configuration models and YAML/JSON serialization for ZeroSwap Engine & Optimizer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import yaml


StrategyVariant = Literal[
    "partial_layer_swap",  # Variant A: Heavy Orchestrator (30B-32B) + Light Guardrail (1B-3B)
    "full_role_pivot",     # Variant B: Light Orchestrator (1B-8B) + Heavy Analyst/Guardrail (14B-32B)
]

RoutingPath = Literal[
    "path_a_pipelined_overwrite",  # Delta N >= Delta N_crit: Full async PCIe overlap (0 ms bubble)
    "path_b_cpu_direct_eager",     # Delta N < Delta N_crit & micro-prompt: AVX-512 CPU evaluation
    "path_c_dynamic_chunking",     # Delta N < Delta N_crit & medium prompt: 1-layer ping-pong micro-chunks
]


@dataclass
class ContextPriorityConfig:
    max_context_length: int = 32768
    kv_cache_dtype: str = "fp8"
    block_size: int = 16


@dataclass
class StreamingConfig:
    enable_one_way_overwrite: bool = True
    enable_double_buffering: bool = True
    pin_host_memory: bool = True
    cpu_fallback_max_tokens: int = 512
    dynamic_chunk_layers: int = 1


@dataclass
class GuardrailConfig:
    enable_pii_sanitizer: bool = True
    enable_task_splitter: bool = True
    max_self_correction_retries: int = 3


@dataclass
class EngineConfig:
    mode: str = "auto"
    target_gpu_memory_utilization: float = 0.92
    context_priority: ContextPriorityConfig = field(default_factory=ContextPriorityConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EngineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        eng = raw.get("engine", raw)
        return cls(
            mode=eng.get("mode", "auto"),
            target_gpu_memory_utilization=float(eng.get("target_gpu_memory_utilization", 0.92)),
            context_priority=ContextPriorityConfig(**eng.get("context_priority", {})),
            streaming=StreamingConfig(**eng.get("streaming", {})),
            guardrail=GuardrailConfig(**eng.get("guardrail", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"engine": asdict(self)}


@dataclass
class ModelTopology:
    model_id: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    bytes_per_param: float
    total_weight_mb: float
    layer_size_mb: float
    kv_bytes_per_token: float


@dataclass
class TunePlan:
    """Auto-generated optimization plan produced by `python -m zeroswap.tune`."""

    gpu_index: int
    gpu_name: str
    total_vram_mb: float
    pcie_bandwidth_gbps: float
    pcie_generation_estimate: str
    primary_model: ModelTopology
    secondary_model: ModelTopology
    strategy_variant: StrategyVariant
    resident_layers: int
    swapped_layers: int
    dynamic_slab_mb: float
    ping_pong_buffer_mb: float
    kv_cache_pool_mb: float
    kv_cache_max_tokens: int
    kv_cache_blocks: int
    single_layer_compute_ms_per_1k_tokens: float
    prefill_throughput_tokens_per_sec: float
    transfer_time_swapped_slab_ms: float
    transfer_time_single_layer_ms: float
    break_even_delta_tokens: int
    degraded_quantization_fallback: Optional[str] = None
    estimated_net_ttft_overhead_ms: float = 0.0

    def to_json(self, path: Optional[str | Path] = None) -> str:
        payload = json.dumps(asdict(self), indent=2)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    @classmethod
    def from_json(cls, path: str | Path) -> "TunePlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["primary_model"] = ModelTopology(**data["primary_model"])
        data["secondary_model"] = ModelTopology(**data["secondary_model"])
        return cls(**data)
