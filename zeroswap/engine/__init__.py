"""Core GPU/PCIe memory and stream orchestration primitives for ZeroSwap."""

from zeroswap.engine.static_slab import KVCacheInvariantError, StaticSlab
from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.layer_interceptor import LayerInterceptor
from zeroswap.engine.tensor_core_pipeline import TensorCoreMetrics, TensorCorePipelineEngine

__all__ = [
    "StaticSlab",
    "KVCacheInvariantError",
    "PinnedHostPool",
    "PCIeStreamer",
    "LayerInterceptor",
    "TensorCorePipelineEngine",
    "TensorCoreMetrics",
]
