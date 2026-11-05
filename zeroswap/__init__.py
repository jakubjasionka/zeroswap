"""ZeroSwap: Zero-Latency In-Flight Weight Swapping & Multiplexing Middleware for vLLM."""

from zeroswap.config import EngineConfig, ModelTopology, TunePlan
from zeroswap.engine.static_slab import KVCacheInvariantError, StaticSlab
from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.layer_interceptor import LayerInterceptor
from zeroswap.optimizer.profiler import AutoProfiler
from zeroswap.optimizer.kv_budget import KVBudgetPlanner
from zeroswap.optimizer.router import RuntimeDecisionRouter
from zeroswap.guardrail.sanitizer import DataSanitizationGate
from zeroswap.guardrail.gate import GuardrailGate, gate

__version__ = "0.1.0"

__all__ = [
    "EngineConfig",
    "ModelTopology",
    "TunePlan",
    "StaticSlab",
    "KVCacheInvariantError",
    "PinnedHostPool",
    "PCIeStreamer",
    "LayerInterceptor",
    "AutoProfiler",
    "KVBudgetPlanner",
    "RuntimeDecisionRouter",
    "DataSanitizationGate",
    "GuardrailGate",
    "gate",
]
