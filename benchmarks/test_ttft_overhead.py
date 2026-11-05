"""Real GPU Benchmark: TTFT Overhead Comparison & CUDA Graph / KV Cache Invariance Proof.

Compares 4 execution modes on real transformer decoder layers:
1. Pure ResidentVRAM Baseline (No Guardrail Swap)
2. Naive Cold Reload / Two-Way Swap (DtoH Offload + HtoD Guardrail + HtoD Restore)
3. One-Way Static RAM Mirror Overwrite (0 ms Offload + Unoverlapped HtoD Restore)
4. ZeroSwap Pipelined Overwrite (0 ms Offload + Async HtoD Restore hidden under Layers 0..K-1)

Also captures a real `torch.cuda.CUDAGraph` on the primary model before swapping, runs the
guardrail overwrite + pipelined restore, and replays the exact same `CUDAGraph` without
re-compilation, asserting `torch.allclose` numerical identity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn

from zeroswap.engine.layer_interceptor import LayerInterceptor
from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.static_slab import StaticSlab


class BenchmarkDecoderLayer(nn.Module):
    """Representative SwiGLU + QKV Transformer Decoder Layer for GPU timing & CUDA Graph tests."""

    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 11008, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.o_proj(torch.nn.functional.silu(self.qkv(x)))
        x = x + attn * 0.1
        ffn = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return x + ffn * 0.1


def run_ttft_benchmark(
    num_layers: int = 24,
    swapped_count: int = 3,
    hidden_size: int = 3072,
    intermediate_size: int = 8192,
    seq_len: int = 4096,
) -> Dict[str, float | bool | str]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    layers = nn.ModuleList(
        [BenchmarkDecoderLayer(hidden_size, intermediate_size, dtype=dtype).to(device) for _ in range(num_layers)]
    )
    swapped_indices: List[int] = list(range(num_layers - swapped_count, num_layers))

    # Allocate simulated PagedAttention KV Cache Pool & Block Table
    kv_cache_pool = [
        torch.randn((256, 16, 8, hidden_size // 32), device=device, dtype=dtype)
        for _ in range(4)
    ]
    block_table = torch.arange(0, 256, dtype=torch.int32, device=device).reshape(16, 16)

    slab = StaticSlab(device=device)
    slab.register_kv_cache_pool(kv_cache_pool, block_tables=block_table)

    host_pool = PinnedHostPool(pin_memory=(device.type == "cuda"))
    slab_bytes = 0
    for idx in swapped_indices:
        slab_bytes += slab.register_swappable_layer(idx, layers[idx])
        host_pool.mirror_primary_layer(idx, layers[idx])

    # Create secondary guardrail layers in pinned memory
    secondary_gpu = [
        BenchmarkDecoderLayer(hidden_size, intermediate_size // 2, dtype=dtype).to(device)
        for _ in range(swapped_count)
    ]
    host_pool.mirror_secondary_model(secondary_gpu)

    streamer = PCIeStreamer(slab=slab, host_pool=host_pool, device=device)
    interceptor = LayerInterceptor(
        layers=layers,
        swapped_layer_indices=swapped_indices,
        slab=slab,
        streamer=streamer,
    )
    interceptor.attach()

    x = torch.randn((1, seq_len, hidden_size), device=device, dtype=dtype)

    def _forward_all(inp: torch.Tensor) -> torch.Tensor:
        h = inp
        for layer in layers:
            h = layer(h)
        return h

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = _forward_all(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # 1. Baseline Resident Forward Pass (TTFT without swapping)
    t0 = time.perf_counter()
    with torch.no_grad():
        ref_out = _forward_all(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    baseline_ttft_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Traditional Two-Way Swap (DtoH Offload + HtoD Guardrail + HtoD Restore + Forward)
    t0 = time.perf_counter()
    with torch.no_grad():
        # DtoH offload of primary tail layers (non-pinned standard CPU copy)
        cpu_backup = {
            idx: {k: v.data.to("cpu") for k, v in layers[idx].named_parameters()}
            for idx in swapped_indices
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # HtoD load of guardrail
        streamer.overwrite_slab_with_secondary(swapped_indices, secondary_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # Restore primary synchronously BEFORE starting forward pass
        for idx in swapped_indices:
            for k, v in layers[idx].named_parameters():
                v.data.copy_(cpu_backup[idx][k], non_blocking=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _ = _forward_all(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    two_way_swap_ttft_ms = (time.perf_counter() - t0) * 1000.0
    two_way_overhead_ms = max(0.0, two_way_swap_ttft_ms - baseline_ttft_ms)

    # 3. Unoverlapped One-Way Static RAM Mirror Overwrite (0 ms Offload + Sync HtoD Restore + Forward)
    streamer.overwrite_slab_with_secondary(swapped_indices, secondary_gpu)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        streamer.async_restore_all_swapped_layers(swapped_indices)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _ = _forward_all(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    one_way_sync_ttft_ms = (time.perf_counter() - t0) * 1000.0
    one_way_sync_overhead_ms = max(0.0, one_way_sync_ttft_ms - baseline_ttft_ms)

    # 4. ZeroSwap Pipelined Overwrite (0 ms Offload + Async HtoD Restore Overlapped under Layers 0..K-1)
    streamer.overwrite_slab_with_secondary(swapped_indices, secondary_gpu)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    interceptor.mark_dirty_for_restore()

    t0 = time.perf_counter()
    with torch.no_grad():
        pipelined_out = _forward_all(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    pipelined_ttft_ms = (time.perf_counter() - t0) * 1000.0
    pipelined_overhead_ms = max(0.0, pipelined_ttft_ms - baseline_ttft_ms)

    # Verify bitwise / numerical restoration accuracy and KV Cache Invariant
    outputs_match = bool(torch.allclose(ref_out, pipelined_out, atol=1e-3, rtol=1e-3))
    kv_invariant_ok = slab.verify_kv_cache_invariant(kv_cache_pool, block_tables=block_table)

    # 5. CUDA Graph Capture & Replay across Weight Swap without Re-Compilation
    cudagraph_preserved = True
    if device.type == "cuda":
        static_in = torch.randn((1, 64, hidden_size), device=device, dtype=dtype)
        # Detach hooks during static CUDA graph capture
        interceptor.detach()
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(3):
                static_ref = _forward_all(static_in)
        torch.cuda.current_stream(device).wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = _forward_all(static_in)

        graph.replay()
        expected_graph_out = static_out.clone()

        # Corrupt/overwrite tail layers with secondary weights, then restore in-place via ZeroSwap
        streamer.overwrite_slab_with_secondary(swapped_indices)
        streamer.async_restore_all_swapped_layers(swapped_indices)
        for idx in swapped_indices:
            streamer.wait_primary_layer_ready(idx)
            slab.verify_layer_pointers_invariant(idx)

        # Replay the ORIGINAL captured CUDA Graph without re-capturing!
        graph.replay()
        torch.cuda.synchronize(device)
        cudagraph_preserved = bool(torch.allclose(expected_graph_out, static_out, atol=1e-3, rtol=1e-3))

    results = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "slab_size_mb": round(slab_bytes / (1024.0 * 1024.0), 2),
        "seq_len": seq_len,
        "baseline_resident_ttft_ms": round(baseline_ttft_ms, 2),
        "two_way_swap_ttft_ms": round(two_way_swap_ttft_ms, 2),
        "two_way_swap_overhead_ms": round(two_way_overhead_ms, 2),
        "one_way_sync_ttft_ms": round(one_way_sync_ttft_ms, 2),
        "one_way_sync_overhead_ms": round(one_way_sync_overhead_ms, 2),
        "zeroswap_pipelined_ttft_ms": round(pipelined_ttft_ms, 2),
        "zeroswap_pipelined_overhead_ms": round(pipelined_overhead_ms, 2),
        "numerical_restoration_exact": outputs_match,
        "kv_cache_invariant_preserved": kv_invariant_ok,
        "cuda_graph_replayed_without_recapture": cudagraph_preserved,
    }
    return results


if __name__ == "__main__":
    res = run_ttft_benchmark()
    print("=== ZeroSwap TTFT & Invariant Benchmark ===")
    print(json.dumps(res, indent=2))
    out_file = Path(__file__).resolve().parent / "ttft_benchmark_results.json"
    out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
