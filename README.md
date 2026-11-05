# ZeroSwap: Zero-Latency In-Flight Weight Swapping & Guardrail Multiplexing for vLLM

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-CUDA_Streams-ee4c2c.svg)](https://pytorch.org/)
[![vLLM Compatible](https://img.shields.io/badge/vLLM-Worker_Extension-00b4d8.svg)](https://github.com/vllm-project/vllm)

**ZeroSwap** is a specialized GPU memory & PCIe DMA middleware for **vLLM** and PyTorch inference engines designed for single-GPU and dual-GPU local workstations (such as **32 GB NVIDIA GeForce RTX 5090**).

It enables running a heavy local **Orchestrator Model** (e.g., 27B–32B Qwen / Nemotron / GLM with a large 32k–128k KV Cache) alongside a local **Decision / Safety / PII Guardrail Model** (e.g., 1B–3B LFM / Qwen3) **without keeping both models in VRAM permanently, without invalidating PagedAttention KV Cache, and without triggering CUDA Graph re-compilation**.

---

## Why ZeroSwap?

On a 32 GB GPU, dedicating 2–6 GB of VRAM 24/7 to a secondary safety/decision model that only runs for 2% of the wall-clock time sacrifices tens of thousands of tokens of PagedAttention KV Cache capacity. Conversely, stopping or swapping model containers from scratch destroys the KV cache and incurs a **30–90 second** CUDA Graph warm-up penalty.

ZeroSwap eliminates both bottlenecks through **4 hardware-level invariants**:

1. **The KV Cache & CUDA Graph Invariant (`StaticSlab`)**:
   PagedAttention block tables and KV cache pools remain 100% untouched in VRAM. Dynamically swapped tail transformer layers are overwritten **in-place** (`tensor.copy_(..., non_blocking=True)`) at fixed virtual GPU addresses (`data_ptr()`). Because tensor addresses, shapes, and strides never mutate, captured `torch.cuda.CUDAGraph` instances replay across swaps with **zero re-compilation**.
2. **One-Way In-Place Overwrite (`PinnedHostPool` — 0 ms Offload)**:
   Inference weights are immutable (`read-only`). ZeroSwap maintains a page-locked (`pin_memory=True` / `cudaHostAlloc`) **Golden Image** of the swapped primary layers and the secondary guardrail weights in system RAM. "Offloading" the primary layers takes **0.00 ms** (zero Device-to-Host PCIe traffic, zero `memset`)—the GPU slab is simply overwritten via asynchronous Host-to-Device (`HtoD`) DMA.
3. **Double-Buffered PCIe Pipelining (`PCIeStreamer` & `LayerInterceptor`)**:
   Uses a high-priority CUDA DMA `Copy Stream` (`priority=-1`) and a `Compute Stream` (`priority=0`) synchronized via `cudaStreamWaitEvent` (`torch.cuda.Event`). When returning from the guardrail to the orchestrator, the orchestrator immediately begins its forward pass on resident layers `0 .. K-1` while the PCIe Copy Engine restores tail layers `K .. N-1` in the background:
   $$T_{\text{transfer}}(\text{Layers } K \dots N-1) \le T_{\text{compute}}(\text{Layers } 0 \dots K-1) \implies \text{Net TTFT Overhead} = \mathbf{0.00\text{ ms}}$$
4. **Zero-Config Auto-Tuner & 3-Path Prefix-Cache-Aware Router (`zeroswap.tune`)**:
   Automatically benchmarks PCIe DMA bandwidth, parses model topologies, solves the **Break-Even Token Delta ($\Delta N_{\text{crit}}$)**, handles model asymmetry (**Partial Layer Swap** vs. **Full Role Pivot**), and routes high-cache-hit micro-deltas dynamically to avoid PCIe bubbles.

---

## Architecture

```
├── zeroswap/
│   ├── engine/
│   │   ├── static_slab.py        # Pre-allocated fixed-pointer GPU slab & KV Cache / CUDA Graph Invariant verifier
│   │   ├── pinned_host.py        # DMA Page-Locked Host Memory Allocator (read-only Golden Image in RAM)
│   │   ├── pcie_streamer.py      # Dual CUDA Streams (High-Priority Copy Stream + Compute Stream) & Event barriers
│   │   └── layer_interceptor.py  # Forward hooks triggering async HtoD layer restoration without breaking CUDA Graphs
│   ├── optimizer/
│   │   ├── profiler.py           # Zero-Config AutoProfiler (PCIe probe, topology parser, Compute probe, ΔN_crit solver)
│   │   ├── kv_budget.py          # Dynamic PagedAttention KV Cache reservation & OOM-free slab/NVFP4 degradation
│   │   └── router.py             # 3-Path Runtime Decision Router (Path A: Pipelined, Path B: CPU AVX-512, Path C: Micro-Chunk)
│   ├── guardrail/
│   │   ├── sanitizer.py          # DataSanitizationGate: Reversible PII/API-key/IP masking + micro-task decomposition
│   │   └── gate.py               # @gate.intercept decorator & Self-Correction Loop with warm KV cache feedback
│   ├── vllm_integration/
│   │   ├── worker_ext.py         # Native vLLM Worker Extension (--worker-extension-cls)
│   │   └── openai_server.py      # OpenAI-compatible Control-Plane Server (/v1/chat/completions, /v1/zeroswap/*)
│   └── tune.py                   # CLI Auto-Tuner (`python -m zeroswap.tune`)
├── benchmarks/
│   ├── test_ttft_overhead.py     # Real GPU TTFT benchmark + CUDA Graph replay & KV Cache invariance proof
│   └── test_safety_loop.py       # End-to-end Orchestrator -> Sanitizer -> Guardrail -> Self-Correction benchmark
├── csrc/
│   └── zeroswap_cuda_ext.cu      # Optional C++/CUDA native bindings for cudaMemcpyAsync
└── tests/                        # Unit test suite (KV invariant, optimizer math, sanitizer, vLLM worker)
```

---

## Real Hardware Benchmark (NVIDIA GeForce RTX 5090 32 GB)

Measured via `python benchmarks/test_ttft_overhead.py` on an **NVIDIA GeForce RTX 5090** (PCIe 4.0 x16 @ 26.4 GB/s pinned HtoD bandwidth, `seq_len=4096`):

| Strategy | Total TTFT (ms) | Net Swap Overhead (ms) | KV Cache Preserved | CUDA Graph Re-Capture Required |
| :--- | :---: | :---: | :---: | :---: |
| **0. Cold Container / Engine Restart** | ~3,500+ ms | +3,400+ ms | ❌ Destroyed | ❌ Full Re-Capture (30–90 s) |
| **1. Baseline (100% Resident in VRAM)** | **87.18 ms** | **0.00 ms** *(ref)* | ✅ Yes | ✅ No |
| **2. Traditional Two-Way Swap (`DtoH + HtoD`)** | 403.22 ms | +316.04 ms | ✅ Yes | ✅ No |
| **3. One-Way Static RAM Mirror (Unoverlapped)** | 105.69 ms | +18.51 ms | ✅ Yes | ✅ No |
| **4. ZeroSwap Pipelined Overwrite (`Copy || Compute`)** | **87.10 ms** | **0.00 ms** | ✅ **100% Intact** | ✅ **Replayed Without Re-Capture** |

---

## Mathematical Model & Dynamic Runtime Routing

### Break-Even Token Delta ($\Delta N_{\text{crit}}$)

In agentic workflows with Prefix Caching (`RadixAttention` in vLLM / SGLang), prefill compute time $T_{\text{compute}}$ scales linearly with the **uncached token delta** $\Delta N = N_{\text{total}} - N_{\text{cached}}$, whereas PCIe transfer time $T_{\text{transfer}}$ depends only on the swapped weight footprint:

$$\Delta N_{\text{crit}} = \left\lceil \frac{\text{Swapped Weight Size (MB)}}{\text{PCIe Bandwidth (MB/s)}} \times \text{Resident Prefill Throughput (tokens/s)} \right\rceil$$

At runtime, `RuntimeDecisionRouter` evaluates $\Delta N$ for every incoming request and selects one of three execution paths:

1. **Path A ($\Delta N \ge \Delta N_{\text{crit}}$) — Pipelined Weight Overwrite**:
   Full asynchronous HtoD restoration overlapped underneath resident layers `0 .. K-1` forward pass (**0.00 ms PCIe bubble**).
2. **Path B ($\Delta N < \Delta N_{\text{crit}}$ & Micro-Prompt $\le 512$ tokens) — Direct CPU AVX-512 Eager**:
   When prefix cache hit rate is ~99% and only a tiny tool diff is appended, the lightweight guardrail runs directly on CPU threads in ~30–45 ms without touching GPU VRAM or the PCIe bus.
3. **Path C ($\Delta N < \Delta N_{\text{crit}}$ & Medium Prompt) — Dynamic 1-Layer Micro-Chunking**:
   Slices the weight swap into 1-layer ping-pong micro-packets (`~10–15 ms` per layer), reducing the effective break-even threshold by a factor of $1 / N_{\text{swapped}}$.

---

## Quick Start

### 1. Installation

```bash
pip install -e ".[dev]"
```

### 2. Run Zero-Config Auto-Tuning Profiler (`zeroswap.tune`)

```bash
python -m zeroswap.tune --primary qwen-32b --secondary lfm-1b --gpu 0
```

Sample output on **NVIDIA GeForce RTX 5090**:
```text
[ZeroSwap] Profiling complete:
  - GPU Device: NVIDIA GeForce RTX 5090 (31.4 GB VRAM)
  - PCIe Bandwidth: 26.4 GB/s (Detected PCIe 4.0 x16)
  - Primary Model (qwen-32b): 64 layers, ~310 MB/layer (19.38 GB total)
  - Guardrail Model (lfm-1b): 1.76 GB total footprint (24 layers)
  - Asymmetry Variant: partial_layer_swap
  - Optimal Strategy: Swap 6 layers (1.82 GB buffer)
  - KV Cache Headroom: 77,872 tokens preserved (4,867 PagedAttention blocks, 9.51 GB pool)
  - Break-Even Token Delta (Delta N_crit): 601 tokens (Prefill: 7,891 tok/s, H2D Slab Transfer: 68.9 ms)
  - Latency impact: 0.00 ms (Fully overlapped during layer 0-57 forward)
```

### 3. Standardized Python Guardrail Decorator (`@gate.intercept`)

```python
from zeroswap import gate

@gate.intercept(safety_model="lfm-1b-guard", mode="pipelined")
async def handle_agent_step(prompt: str, kv_context: dict) -> str:
    # 1. PII, API keys, and internal IPs are automatically replaced with synthetic mocks (<MOCK_API_KEY_1>)
    # 2. Guardrail verifies the step in-flight using ZeroSwap 3-path routing
    # 3. If rejected, Self-Correction Loop injects feedback into warm KV cache without losing context
    # 4. Synthetic mocks are automatically restored back to real values on return
    return f"Executed agent step on sanitized payload: {prompt}"
```

### 4. Integration with vLLM (`ZeroSwapWorkerExtension` & OpenAI Control Plane)

**Option A — Native vLLM Worker Extension (`--worker-extension-cls`)**:
```bash
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --worker-extension-cls zeroswap.vllm_integration.ZeroSwapWorkerExtension \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching
```

**Option B — OpenAI-Compatible Control-Plane Interceptor Server**:
```bash
python -m zeroswap.vllm_integration.openai_server \
  --port 8080 \
  --upstream-vllm http://localhost:8000 \
  --primary qwen-32b \
  --secondary lfm-1b
```

---

## Running Tests & Benchmarks

```bash
# Run unit test suite
pytest -v

# Run real GPU TTFT + CUDA Graph + KV Cache Invariant benchmark
python benchmarks/test_ttft_overhead.py

# Run PII Sanitization + Guardrail Self-Correction Loop benchmark
python benchmarks/test_safety_loop.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
