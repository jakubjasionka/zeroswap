"""PinnedHostPool: Page-locked DMA Host Memory Allocator (`pin_memory=True` / `cudaHostAlloc`).

Maintains the read-only Golden Image of both the Primary Orchestrator's swapped layers
and the Secondary Guardrail model in page-locked system RAM so the PCIe DMA engine can
stream HtoD transfers at full link bandwidth (~25 GB/s PCIe 4.0 / ~50 GB/s PCIe 5.0)
with zero CPU paging overhead and zero DtoH offload cost.
"""

from __future__ import annotations

from typing import Dict, List
import torch
import torch.nn as nn


class PinnedHostPool:
    """Manages page-locked (pinned) host memory mirrors for primary and secondary model layers."""

    def __init__(self, pin_memory: bool = True) -> None:
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.primary_golden_layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.secondary_golden_layers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.total_pinned_bytes: int = 0

    def _allocate_pinned_copy(self, tensor: torch.Tensor) -> torch.Tensor:
        host_tensor = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        host_tensor.copy_(tensor.detach(), non_blocking=False)
        self.total_pinned_bytes += host_tensor.numel() * host_tensor.element_size()
        return host_tensor

    def mirror_primary_layer(self, layer_idx: int, module: nn.Module) -> int:
        """Creates an immutable page-locked RAM Golden Image of a primary model layer."""
        layer_mirror: Dict[str, torch.Tensor] = {}
        layer_bytes = 0
        for name, param in module.named_parameters(recurse=True):
            pinned = self._allocate_pinned_copy(param.data)
            layer_mirror[name] = pinned
            layer_bytes += pinned.numel() * pinned.element_size()
        for name, buf in module.named_buffers(recurse=True):
            pinned = self._allocate_pinned_copy(buf)
            layer_mirror[f"buf:{name}"] = pinned
            layer_bytes += pinned.numel() * pinned.element_size()
        self.primary_golden_layers[layer_idx] = layer_mirror
        return layer_bytes

    def mirror_secondary_model(self, layers: nn.ModuleList | List[nn.Module]) -> int:
        """Mirrors all layers of the secondary guardrail model into page-locked RAM."""
        total_bytes = 0
        for idx, layer in enumerate(layers):
            layer_mirror: Dict[str, torch.Tensor] = {}
            for name, param in layer.named_parameters(recurse=True):
                pinned = self._allocate_pinned_copy(param.data)
                layer_mirror[name] = pinned
                total_bytes += pinned.numel() * pinned.element_size()
            for name, buf in layer.named_buffers(recurse=True):
                pinned = self._allocate_pinned_copy(buf)
                layer_mirror[f"buf:{name}"] = pinned
                total_bytes += pinned.numel() * pinned.element_size()
            self.secondary_golden_layers[idx] = layer_mirror
        return total_bytes

    def get_primary_layer_mirror(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        if layer_idx not in self.primary_golden_layers:
            raise KeyError(f"Primary layer {layer_idx} not found in PinnedHostPool.")
        return self.primary_golden_layers[layer_idx]

    def get_secondary_layer_mirror(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        if layer_idx not in self.secondary_golden_layers:
            raise KeyError(f"Secondary layer {layer_idx} not found in PinnedHostPool.")
        return self.secondary_golden_layers[layer_idx]
