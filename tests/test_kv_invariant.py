"""Unit tests for StaticSlab, PinnedHostPool, PCIeStreamer, LayerInterceptor, and KV Cache Invariant."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from zeroswap.engine.layer_interceptor import LayerInterceptor
from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.static_slab import KVCacheInvariantError, StaticSlab


class TinyTransformerBlock(nn.Module):
    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2, bias=False)
        self.fc2 = nn.Linear(d * 2, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.relu(self.fc1(x)))


def test_kv_cache_and_layer_pointer_invariance() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layers = nn.ModuleList([TinyTransformerBlock(128).to(device) for _ in range(6)])
    swapped_indices = [4, 5]

    kv_caches = [torch.randn((32, 16, 4, 32), device=device) for _ in range(2)]
    block_tables = torch.arange(0, 32, dtype=torch.int32, device=device).reshape(4, 8)

    slab = StaticSlab(device=device, ping_pong_layer_bytes=65536)
    slab.register_kv_cache_pool(kv_caches, block_tables=block_tables)

    host_pool = PinnedHostPool(pin_memory=(device.type == "cuda"))
    for idx in swapped_indices:
        slab.register_swappable_layer(idx, layers[idx])
        host_pool.mirror_primary_layer(idx, layers[idx])

    secondary_layers = [TinyTransformerBlock(128).to(device) for _ in range(2)]
    host_pool.mirror_secondary_model(secondary_layers)

    streamer = PCIeStreamer(slab=slab, host_pool=host_pool, device=device)
    interceptor = LayerInterceptor(
        layers=layers,
        swapped_layer_indices=swapped_indices,
        slab=slab,
        streamer=streamer,
    )
    interceptor.attach()

    x = torch.randn((2, 16, 128), device=device)
    with torch.no_grad():
        expected = x
        for layer in layers:
            expected = layer(expected)

    # Record original pointers
    orig_ptrs = {
        idx: {k: p.data_ptr() for k, p in layers[idx].named_parameters()}
        for idx in swapped_indices
    }

    # Overwrite with secondary guardrail weights (0 ms DtoH)
    streamer.overwrite_slab_with_secondary(swapped_indices, secondary_layers)
    interceptor.mark_dirty_for_restore()

    # Run primary forward pass -> Layer 0 hook triggers async HtoD restore of layers 4, 5
    with torch.no_grad():
        actual = x
        for layer in layers:
            actual = layer(actual)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    assert torch.allclose(expected, actual, atol=1e-5)
    assert slab.verify_kv_cache_invariant(kv_caches, block_tables=block_tables)

    for idx in swapped_indices:
        assert slab.verify_layer_pointers_invariant(idx)
        for k, p in layers[idx].named_parameters():
            assert p.data_ptr() == orig_ptrs[idx][k]

    interceptor.detach()


def test_kv_invariant_detects_block_table_corruption() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kv_caches = [torch.zeros((8, 16, 4, 16), device=device)]
    block_tables = torch.arange(0, 16, dtype=torch.int32, device=device)

    slab = StaticSlab(device=device)
    slab.register_kv_cache_pool(kv_caches, block_tables=block_tables)

    assert slab.verify_kv_cache_invariant(kv_caches, block_tables=block_tables)

    # Corrupt block table in-place
    block_tables[0] += 99
    with pytest.raises(KVCacheInvariantError):
        slab.verify_kv_cache_invariant(kv_caches, block_tables=block_tables)


def test_ping_pong_secondary_streaming() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sec_layers = nn.ModuleList([TinyTransformerBlock(64).to(device) for _ in range(4)])
    x = torch.randn((1, 8, 64), device=device)

    with torch.no_grad():
        expected = x
        for layer in sec_layers:
            expected = layer(expected)

    slab = StaticSlab(device=device)
    host_pool = PinnedHostPool(pin_memory=(device.type == "cuda"))
    host_pool.mirror_secondary_model(sec_layers)
    streamer = PCIeStreamer(slab=slab, host_pool=host_pool, device=device)

    template = TinyTransformerBlock(64).to(device)
    with torch.no_grad():
        pipelined_out = streamer.pipelined_secondary_forward(
            layer_template=template,
            num_secondary_layers=len(sec_layers),
            hidden_states=x,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    assert torch.allclose(expected, pipelined_out, atol=1e-5)
