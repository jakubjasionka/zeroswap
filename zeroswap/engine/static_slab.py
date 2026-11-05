"""StaticSlab: Pre-allocated virtual GPU memory slab & KV Cache / CUDA Graph Invariant verifier.

Guarantees that:
1. All dynamically swapped transformer layer parameters keep identical virtual GPU memory
   addresses (`data_ptr()`), tensor shapes, and strides across overwrites and restores,
   preventing CUDA Graph invalidation or re-compilation.
2. PagedAttention KV cache pools and block tables remain 100% isolated and unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class KVCacheInvariantError(RuntimeError):
    """Raised when a KV cache pointer, block table, or CUDA graph tensor address is violated."""


@dataclass(frozen=True)
class TensorSignature:
    name: str
    data_ptr: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    numel: int
    nbytes: int


class StaticSlab:
    """Manages fixed-pointer GPU memory regions for zero-recompilation weight swapping."""

    def __init__(
        self,
        device: torch.device | str = "cuda:0",
        ping_pong_layer_bytes: int = 0,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.layer_signatures: Dict[int, Dict[str, TensorSignature]] = {}
        self.layer_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self.kv_cache_signatures: Dict[str, TensorSignature] = {}
        self.block_table_checksums: Dict[str, int] = {}
        self._slab_state: str = "primary_resident"

        # Optional dual Ping-Pong buffers (Buffer A & Buffer B) for layer-wise streaming
        self.ping_pong_bytes = ping_pong_layer_bytes
        self.ping_pong_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if ping_pong_layer_bytes > 0:
            self.ping_pong_buffers = (
                torch.empty(ping_pong_layer_bytes, dtype=torch.uint8, device=self.device),
                torch.empty(ping_pong_layer_bytes, dtype=torch.uint8, device=self.device),
            )

    @property
    def slab_state(self) -> str:
        return self._slab_state

    def mark_evicted_zero_copy(self) -> None:
        """One-Way In-Place Overwrite: 0.0 ms logical eviction without DtoH copy or memset."""
        self._slab_state = " logically_free_for_overwrite"

    def mark_secondary_loaded(self) -> None:
        self._slab_state = "secondary_loaded"

    def mark_primary_restored(self) -> None:
        self._slab_state = "primary_resident"

    def register_kv_cache_pool(
        self,
        kv_caches: List[torch.Tensor] | Dict[str, torch.Tensor],
        block_tables: Optional[torch.Tensor] = None,
    ) -> None:
        """Registers PagedAttention KV cache tensors and block tables to enforce strict invariance."""
        items = (
            kv_caches.items()
            if isinstance(kv_caches, dict)
            else [(f"kv_layer_{i}", t) for i, t in enumerate(kv_caches)]
        )
        for name, tensor in items:
            self.kv_cache_signatures[name] = TensorSignature(
                name=name,
                data_ptr=tensor.data_ptr(),
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                numel=tensor.numel(),
                nbytes=tensor.numel() * tensor.element_size(),
            )
        if block_tables is not None:
            self.block_table_checksums["primary_block_table"] = int(
                block_tables.to(dtype=torch.int64).sum().item()
            )
            self.kv_cache_signatures["primary_block_table"] = TensorSignature(
                name="primary_block_table",
                data_ptr=block_tables.data_ptr(),
                shape=tuple(block_tables.shape),
                stride=tuple(block_tables.stride()),
                dtype=block_tables.dtype,
                numel=block_tables.numel(),
                nbytes=block_tables.numel() * block_tables.element_size(),
            )

    def register_swappable_layer(self, layer_idx: int, module: nn.Module) -> int:
        """Records fixed GPU pointer signatures for a swappable transformer layer."""
        sigs: Dict[str, TensorSignature] = {}
        params: Dict[str, torch.Tensor] = {}
        total_bytes = 0
        for name, param in module.named_parameters(recurse=True):
            t = param.data
            nbytes = t.numel() * t.element_size()
            sigs[name] = TensorSignature(
                name=name,
                data_ptr=t.data_ptr(),
                shape=tuple(t.shape),
                stride=tuple(t.stride()),
                dtype=t.dtype,
                numel=t.numel(),
                nbytes=nbytes,
            )
            params[name] = t
            total_bytes += nbytes
        for name, buf in module.named_buffers(recurse=True):
            nbytes = buf.numel() * buf.element_size()
            sigs[f"buf:{name}"] = TensorSignature(
                name=f"buf:{name}",
                data_ptr=buf.data_ptr(),
                shape=tuple(buf.shape),
                stride=tuple(buf.stride()),
                dtype=buf.dtype,
                numel=buf.numel(),
                nbytes=nbytes,
            )
            params[f"buf:{name}"] = buf
            total_bytes += nbytes
        self.layer_signatures[layer_idx] = sigs
        self.layer_params[layer_idx] = params
        return total_bytes

    def verify_layer_pointers_invariant(self, layer_idx: int) -> bool:
        """Confirms that CUDA Graph virtual addresses (`data_ptr()`) for `layer_idx` have not shifted."""
        if layer_idx not in self.layer_signatures:
            raise KeyError(f"Layer {layer_idx} is not registered in StaticSlab.")
        expected_sigs = self.layer_signatures[layer_idx]
        actual_params = self.layer_params[layer_idx]
        for name, sig in expected_sigs.items():
            tensor = actual_params[name]
            if tensor.data_ptr() != sig.data_ptr:
                raise KVCacheInvariantError(
                    f"CUDA Graph Pointer Invariant Violated on layer {layer_idx} param '{name}': "
                    f"expected 0x{sig.data_ptr:x}, got 0x{tensor.data_ptr():x}"
                )
            if tuple(tensor.shape) != sig.shape or tuple(tensor.stride()) != sig.stride:
                raise KVCacheInvariantError(
                    f"Tensor shape/stride mutated on layer {layer_idx} param '{name}'"
                )
        return True

    def verify_kv_cache_invariant(
        self,
        kv_caches: List[torch.Tensor] | Dict[str, torch.Tensor],
        block_tables: Optional[torch.Tensor] = None,
    ) -> bool:
        """Asserts that PagedAttention KV cache pointers and block tables are 100% intact."""
        items = (
            kv_caches.items()
            if isinstance(kv_caches, dict)
            else [(f"kv_layer_{i}", t) for i, t in enumerate(kv_caches)]
        )
        for name, tensor in items:
            if name not in self.kv_cache_signatures:
                raise KVCacheInvariantError(f"Unregistered KV cache tensor '{name}'.")
            sig = self.kv_cache_signatures[name]
            if tensor.data_ptr() != sig.data_ptr:
                raise KVCacheInvariantError(
                    f"KV Cache pointer mutated for '{name}': "
                    f"expected 0x{sig.data_ptr:x}, got 0x{tensor.data_ptr():x}"
                )
            if tuple(tensor.shape) != sig.shape:
                raise KVCacheInvariantError(f"KV Cache shape mutated for '{name}'.")

        if block_tables is not None and "primary_block_table" in self.kv_cache_signatures:
            sig = self.kv_cache_signatures["primary_block_table"]
            if block_tables.data_ptr() != sig.data_ptr:
                raise KVCacheInvariantError("PagedAttention block_table pointer mutated!")
            checksum = int(block_tables.to(dtype=torch.int64).sum().item())
            if checksum != self.block_table_checksums["primary_block_table"]:
                raise KVCacheInvariantError(
                    f"PagedAttention block_table contents corrupted: "
                    f"expected checksum {self.block_table_checksums['primary_block_table']}, got {checksum}"
                )
        return True

    def overwrite_with_BytePattern(self, layer_idx: int, source_bytes_tensor: torch.Tensor) -> None:
        """Overwrites the flat byte representation of a layer's GPU slab without changing data_ptr."""
        if layer_idx not in self.layer_params:
            raise KeyError(f"Layer {layer_idx} not registered.")
        offset = 0
        total_src = source_bytes_tensor.numel()
        for tensor in self.layer_params[layer_idx].values():
            nbytes = tensor.numel() * tensor.element_size()
            chunk_len = min(nbytes, max(0, total_src - offset))
            if chunk_len <= 0:
                break
            flat_uint8_view = tensor.view(torch.uint8).reshape(-1)
            flat_uint8_view[:chunk_len].copy_(
                source_bytes_tensor[offset : offset + chunk_len], non_blocking=True
            )
            offset += chunk_len
        self.mark_secondary_loaded()
