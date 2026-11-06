"""ZeroSwap: Zero-Latency In-Flight Weight Swapping & Multiplexing Middleware for vLLM."""

from zeroswap.config import EngineConfig, ModelTopology, TunePlan
from zeroswap.engine.static_slab import KVCacheInvariantError, StaticSlab
from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.layer_interceptor import LayerInterceptor
from zeroswap.engine.tensor_core_pipeline import TensorCoreMetrics, TensorCorePipelineEngine
from zeroswap.optimizer.profiler import AutoProfiler
from zeroswap.optimizer.kv_budget import KVBudgetPlanner
from zeroswap.optimizer.router import RuntimeDecisionRouter
from zeroswap.guardrail.sanitizer import DataSanitizationGate
from zeroswap.guardrail.gate import GuardrailGate, gate
from zeroswap.octop import (
    ExternalAPISimulator,
    LiquidD1DecisionEngine,
    OctopControlPlane,
    OctopDelegationResult,
    OctopSafetyDecision,
)

__version__ = "0.2.0"

__all__ = [
    "EngineConfig",
    "ModelTopology",
    "TunePlan",
    "StaticSlab",
    "KVCacheInvariantError",
    "PinnedHostPool",
    "PCIeStreamer",
    "LayerInterceptor",
    "TensorCorePipelineEngine",
    "TensorCoreMetrics",
    "AutoProfiler",
    "KVBudgetPlanner",
    "RuntimeDecisionRouter",
    "DataSanitizationGate",
    "GuardrailGate",
    "gate",
    "LiquidD1DecisionEngine",
    "OctopSafetyDecision",
    "OctopControlPlane",
    "OctopDelegationResult",
    "ExternalAPISimulator",
]
