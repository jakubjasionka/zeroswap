"""PCIeStreamer: Dual CUDA Streams (High-Priority Copy Stream + Compute Stream) & Event Barriers.

Implements:
1. One-Way HtoD In-Place Overwrite via native C++/CUDA extension `zeroswap_cuda_ext`
   (`async_batch_inplace_h2d_overwrite` / `cudaMemcpyAsync`) with PyTorch fallback.
2. Event-synchronized latency hiding (`cudaStreamWaitEvent`) so PCIe transfers execute
   underneath the resident layers' forward pass.
3. Ping-Pong Double Buffering (2 slots in GPU memory) for streaming secondary guardrail layers
   or dynamic micro-chunks (`Path C`).
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional
import torch
import torch.nn as nn

from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.static_slab import StaticSlab, iter_leaf_tensors

try:
    import zeroswap_cuda_ext as _cuda_ext
    HAS_NATIVE_CUDA_EXT = True
except ImportError:
    try:
        from zeroswap.engine import zeroswap_cuda_ext as _cuda_ext  # type: ignore[no-redef]
        HAS_NATIVE_CUDA_EXT = True
    except ImportError:
        _cuda_ext = None
        HAS_NATIVE_CUDA_EXT = False


class PCIeStreamer:
    """Orchestrates overlapping PCIe DMA transfers and SM compute via CUDA Streams & Events."""

    def __init__(
        self,
        slab: StaticSlab,
        host_pool: PinnedHostPool,
        device: torch.device | str = "cuda:0",
    ) -> None:
        self.slab = slab
        self.host_pool = host_pool
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.use_native_cuda_ext = bool(self.use_cuda and HAS_NATIVE_CUDA_EXT)

        if self.use_cuda:
            # High-priority DMA Copy Stream (-1) and standard Compute Stream (0)
            self.copy_stream = torch.cuda.Stream(device=self.device, priority=-1)
            self.compute_stream = torch.cuda.Stream(device=self.device, priority=0)
        else:
            self.copy_stream = None
            self.compute_stream = None

        self.primary_restore_events: Dict[int, torch.cuda.Event] = {}
        self.secondary_load_events: Dict[int, torch.cuda.Event] = {}
        self.total_bytes_streamed_h2d: int = 0

    def _copy_layer_dict_inplace(
        self,
        target_params: Dict[str, torch.Tensor],
        source_pinned: Dict[str, torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None,
    ) -> int:
        if (
            self.use_native_cuda_ext
            and _cuda_ext is not None
            and stream is not None
        ):
            dst_list: List[torch.Tensor] = []
            src_list: List[torch.Tensor] = []
            for name, dst_tensor in target_params.items():
                if name in source_pinned:
                    src_tensor = source_pinned[name]
                    if dst_tensor.is_cuda and not src_tensor.is_cuda and dst_tensor.is_contiguous() and src_tensor.is_contiguous():
                        dst_list.append(dst_tensor)
                        src_list.append(src_tensor)
                    else:
                        dst_tensor.copy_(src_tensor, non_blocking=self.use_cuda)
            if dst_list:
                copied = int(
                    _cuda_ext.async_batch_inplace_h2d_overwrite(
                        dst_list, src_list, int(stream.cuda_stream)
                    )
                )
                self.total_bytes_streamed_h2d += copied
                return copied

        copied_bytes = 0
        for name, dst_tensor in target_params.items():
            if name in source_pinned:
                dst_tensor.copy_(source_pinned[name], non_blocking=self.use_cuda)
                copied_bytes += dst_tensor.numel() * dst_tensor.element_size()
        self.total_bytes_streamed_h2d += copied_bytes
        return copied_bytes

    def async_restore_primary_layer(self, layer_idx: int) -> None:
        """Asynchronously copies a primary layer's Golden Image from Pinned RAM to its StaticSlab GPU address."""
        target_params = self.slab.layer_params[layer_idx]
        source_pinned = self.host_pool.get_primary_layer_mirror(layer_idx)

        if self.use_cuda and self.copy_stream is not None:
            # Ensure copy_stream waits for any pending compute on the slab before overwriting
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.copy_stream):
                self._copy_layer_dict_inplace(target_params, source_pinned, stream=self.copy_stream)
                ev = torch.cuda.Event(enable_timing=False)
                ev.record(self.copy_stream)
                self.primary_restore_events[layer_idx] = ev
        else:
            self._copy_layer_dict_inplace(target_params, source_pinned)

    def async_restore_all_swapped_layers(self, swapped_layer_indices: List[int]) -> None:
        """Queues non-blocking HtoD restoration for all swapped tail layers in topological order."""
        for idx in swapped_layer_indices:
            self.async_restore_primary_layer(idx)
        self.slab.mark_primary_restored()

    def wait_primary_layer_ready(self, layer_idx: int) -> None:
        """Blocks the compute stream via `cudaStreamWaitEvent` ONLY if DMA for `layer_idx` is still in flight."""
        if self.use_cuda and layer_idx in self.primary_restore_events:
            ev = self.primary_restore_events[layer_idx]
            torch.cuda.current_stream(self.device).wait_event(ev)

    def extract_module_leaf_map(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """Extracts leaf tensor map (including torchao `Int8Tensor` `qdata`/`scale` leaves) for a module."""
        leaf_map: Dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters(recurse=True):
            for leaf_name, t in iter_leaf_tensors(name, param):
                leaf_map[leaf_name] = t
        for name, buf in module.named_buffers(recurse=True):
            for leaf_name, t in iter_leaf_tensors(f"buf:{name}", buf):
                leaf_map[leaf_name] = t
        return leaf_map

    def overwrite_slab_with_secondary(
        self,
        target_layer_indices: List[int],
        secondary_gpu_modules: Optional[List[nn.Module] | nn.ModuleList] = None,
    ) -> None:
        """One-Way In-Place Overwrite: 0 ms offload of primary layers + async HtoD load of secondary weights."""
        # Step 1: Logical 0 ms eviction (no DtoH transfer needed since Golden Image lives in Pinned RAM)
        self.slab.mark_evicted_zero_copy()

        # Step 2: Stream secondary weights into the GPU slab slots
        if self.use_cuda and self.copy_stream is not None:
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.copy_stream):
                num_slots = (
                    max(len(target_layer_indices), len(secondary_gpu_modules))
                    if secondary_gpu_modules is not None
                    else len(target_layer_indices)
                )
                for slot_idx in range(num_slots):
                    if slot_idx in self.host_pool.secondary_golden_layers:
                        sec_mirror = self.host_pool.get_secondary_layer_mirror(slot_idx)
                        if secondary_gpu_modules is not None and slot_idx < len(secondary_gpu_modules):
                            dst_map = self.extract_module_leaf_map(secondary_gpu_modules[slot_idx])
                            self._copy_layer_dict_inplace(dst_map, sec_mirror, stream=self.copy_stream)
                        elif slot_idx < len(target_layer_indices):
                            layer_idx = target_layer_indices[slot_idx]
                            flat_chunks = [
                                t.reshape(-1).view(torch.uint8) for t in sec_mirror.values()
                            ]
                            if flat_chunks:
                                combined = torch.cat(flat_chunks, dim=0)
                                self.slab.overwrite_with_BytePattern(layer_idx, combined)
                        ev = torch.cuda.Event(enable_timing=False)
                        ev.record(self.copy_stream)
                        self.secondary_load_events[slot_idx] = ev
        self.slab.mark_secondary_loaded()

    def wait_secondary_ready(self) -> None:
        """Synchronizes compute stream with all in-flight secondary layer DMA transfers."""
        if self.use_cuda:
            cur_stream = torch.cuda.current_stream(self.device)
            for ev in self.secondary_load_events.values():
                cur_stream.wait_event(ev)

    def pipelined_secondary_forward(
        self,
        layer_template: nn.Module,
        num_secondary_layers: int,
        hidden_states: torch.Tensor,
        forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Executes secondary guardrail layers using Ping-Pong Double Buffering (2 GPU layer buffers).

        Slot A (index 0) computes Layer `i` on the Compute Stream while
        Slot B (index 1) asynchronously prefetches Layer `i + 1` on the DMA Copy Stream.
        """
        slot_a = copy.deepcopy(layer_template).to(self.device)
        slot_b = copy.deepcopy(layer_template).to(self.device)
        slots = [slot_a, slot_b]
        slot_params = [self.extract_module_leaf_map(slot) for slot in slots]

        # Mark primary slab logically overwritten (0 ms DtoH)
        self.slab.mark_evicted_zero_copy()

        events: Dict[int, torch.cuda.Event] = {}
        compute_done_events: Dict[int, torch.cuda.Event] = {}

        def _enqueue_load(layer_idx: int, slot_id: int) -> None:
            mirror = self.host_pool.get_secondary_layer_mirror(layer_idx)
            if self.use_cuda and self.copy_stream is not None:
                if slot_id in compute_done_events:
                    self.copy_stream.wait_event(compute_done_events[slot_id])
                with torch.cuda.stream(self.copy_stream):
                    self._copy_layer_dict_inplace(slot_params[slot_id], mirror, stream=self.copy_stream)
                    ev = torch.cuda.Event(enable_timing=False)
                    ev.record(self.copy_stream)
                    events[layer_idx] = ev
            else:
                self._copy_layer_dict_inplace(slot_params[slot_id], mirror)

        # Prefetch Layer 0 into Slot A
        if num_secondary_layers > 0:
            _enqueue_load(0, 0)

        x = hidden_states
        for i in range(num_secondary_layers):
            cur_slot = i % 2
            next_layer = i + 1
            next_slot = next_layer % 2

            # Overlap: trigger DMA HtoD for Layer i+1 into the alternate slot BEFORE computing Layer i
            if next_layer < num_secondary_layers:
                _enqueue_load(next_layer, next_slot)

            # Wait for Layer i DMA to complete on the compute stream
            if self.use_cuda and i in events:
                torch.cuda.current_stream(self.device).wait_event(events[i])

            # Compute Layer i in Eager mode (no torch.compile overhead for single-prefill guardrail)
            if forward_fn is not None:
                x = forward_fn(slots[cur_slot], x)
            else:
                out = slots[cur_slot](x)
                x = out[0] if isinstance(out, tuple) else out

            if self.use_cuda:
                done_ev = torch.cuda.Event(enable_timing=False)
                done_ev.record(torch.cuda.current_stream(self.device))
                compute_done_events[cur_slot] = done_ev

        self.slab.mark_secondary_loaded()
        return x
