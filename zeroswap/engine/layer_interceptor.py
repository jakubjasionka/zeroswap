"""LayerInterceptor: Zero-overhead forward hooks for asynchronous layer restoration in PyTorch & vLLM.

Hooks directly into the primary orchestrator's decoder layer list (`nn.ModuleList`):
- When the orchestrator starts its forward pass after a guardrail check, Layer 0's pre-hook
  immediately triggers non-blocking HtoD DMA restoration of swapped tail layers `[K .. N-1]`
  on `copy_stream`.
- Resident layers `[0 .. K-1]` execute immediately at full VRAM bandwidth with zero delay.
- Swapped tail layers `[K .. N-1]` execute `wait_primary_layer_ready(layer_idx)` in their
  pre-hook (`cudaStreamWaitEvent`), which resolves in ~0.00 ms whenever
  `T_transfer(K..N-1) <= T_compute(0..K-1)`.
"""

from __future__ import annotations

from typing import Any, List
import torch.nn as nn

from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.static_slab import StaticSlab


class LayerInterceptor:
    """Attaches forward pre-hooks to transformer layers for pipelined in-flight weight restoration."""

    def __init__(
        self,
        layers: nn.ModuleList | List[nn.Module],
        swapped_layer_indices: List[int],
        slab: StaticSlab,
        streamer: PCIeStreamer,
    ) -> None:
        self.layers = layers
        self.swapped_layer_indices = sorted(swapped_layer_indices)
        self.slab = slab
        self.streamer = streamer
        self._handles: List[Any] = []
        self._needs_restore: bool = False

    def mark_dirty_for_restore(self) -> None:
        """Called after guardrail verification finishes so the next orchestrator forward pass auto-restores."""
        self._needs_restore = True

    def attach(self) -> None:
        """Registers lightweight forward pre-hooks on Layer 0 (trigger) and Swapped Layers (wait barriers)."""
        self.detach()
        if not self.swapped_layer_indices:
            return

        # Hook on Layer 0: Kick off async HtoD DMA restoration of tail layers while Layer 0..K-1 compute
        def _layer0_trigger_hook(module: nn.Module, args: Any) -> None:
            if self._needs_restore:
                self.streamer.async_restore_all_swapped_layers(self.swapped_layer_indices)
                self._needs_restore = False

        h0 = self.layers[0].register_forward_pre_hook(_layer0_trigger_hook)
        self._handles.append(h0)

        # Hook on each swapped layer k in [K..N-1]: wait on its specific CUDA Event barrier
        for idx in self.swapped_layer_indices:
            def _make_wait_hook(layer_idx: int):
                def _wait_hook(module: nn.Module, args: Any) -> None:
                    self.streamer.wait_primary_layer_ready(layer_idx)
                    self.slab.verify_layer_pointers_invariant(layer_idx)
                return _wait_hook

            hk = self.layers[idx].register_forward_pre_hook(_make_wait_hook(idx))
            self._handles.append(hk)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
