# ZeroSwap: Zero-Latency In-Flight Weight Swapping & Octop Guardrail Multiplexing for vLLM

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA C++ Ext](https://img.shields.io/badge/CUDA_C++-sm__120_Blackwell-76b900.svg)](csrc/zeroswap_cuda_ext.cu)
[![vLLM Compatible](https://img.shields.io/badge/vLLM-Worker_Extension-00b4d8.svg)](https://github.com/vllm-project/vllm)

**ZeroSwap** is a specialized GPU memory, native C++/CUDA (`zeroswap_cuda_ext`), and PCIe DMA middleware for **vLLM** and PyTorch inference engines designed for single-GPU and dual-GPU local workstations (such as **32 GB NVIDIA GeForce RTX 5090**).

It enables co-orchestrating a heavy local **Primary Model** (e.g., **`nvidia/Qwen3.8-27B-NVFP4`** — 64 layers, 20.42 GiB weights + 7.65–8.47 GB PagedAttention FP8 KV Cache) alongside a real **System-One Decision / Safety Guardrail Model** (e.g., **`LiquidAI/d1-3B-w8a8`** — 30 hybrid INT8 W8A8 layers, 3.74 GB) **without keeping both models in VRAM permanently, without invalidating PagedAttention KV Cache, and without triggering CUDA Graph re-compilation**.

---

## Octop Hybrid Control Plane & External API Retention Safety

In the **Octop Hybrid Architecture**, the local GPU runs an uncensored, private orchestrator (`nvidia/Qwen3.8-27B-NVFP4`) that delegates heavy coding or analytical subtasks to commercial External Frontier APIs (Claude Code, Cursor, OpenAI, etc.) under the **strict assumption that the external API permanently retains and trains on all submitted prompts**.

Before any subtask leaves the local node:
1. **0.00 ms One-Way Eviction**: ZeroSwap logically evicts the tail layers (`layers 49..63`, 3.77 GB) of `Qwen3.8-27B-NVFP4` in 0.00 ms (no `DtoH` copy because a read-only Golden Image resides in Page-Locked Host RAM).
2. **Single-Pass `LiquidAI/d1-3B-w8a8` Evaluation (`output_tokens = 0`)**: The INT8 W8A8 decision model evaluates the candidate prompt in **one forward pass with zero generated tokens** (`model.system_one`) across 4 calibrated axes:
   - `safe_for_external_retention` (`noul` probability $P(\text{safe})$)
   - `contains_censorship_triggers` (`noul` probability $P(\text{censorship})$)
   - `delegation_decision` (`choice`: `delegate_to_external_api` | `sanitize_and_decompose_microtasks` | `reject_force_local_execution`)
   - `retention_exposure_severity` (`score`: `0.0` to `3.0`)
3. **Sanitization, Anti-Censorship Micro-Task Decomposition & Self-Correction Loop**:
   - **PII / Credentials / Internal IPs** are masked with deterministic reversible tokens (`<MOCK_API_KEY_1>`, `<MOCK_INTERNAL_IP_1>`, `<MOCK_PESEL_OR_SSN_1>`) and unmasked locally when the external API responds.
   - **Security / Reverse-Engineering / Censorship-Sensitive Queries** are decomposed into neutral algorithmic micro-tasks (`[MicroTask #1 - Pure Algorithmic Specification]`) and re-verified by `LiquidAI/d1-3B-w8a8`.
   - **Critical Private Keys / Destructive Payloads** are hard-blocked (`reject_force_local_execution`) and routed into the **Self-Correction Loop** with `Qwen3.8-27B-NVFP4`'s hot KV Cache intact.
4. **Native C++/CUDA Pipelined Restoration (`zeroswap_cuda_ext`)**:
   As `Qwen3.8-27B-NVFP4` resumes execution on its resident layers `0..48`, `zeroswap_cuda_ext.async_batch_inplace_h2d_overwrite` streams the 15 swapped NVFP4/FP8 tail layers (`3,773 MB`) back from Pinned Host RAM into their exact original GPU `data_ptr()` addresses on a high-priority CUDA DMA stream (`priority=-1`) synchronized via `cudaStreamWaitEvent`.

---

## Real Hardware Benchmarks on NVIDIA GeForce RTX 5090 (32 GB)

### 1. `nvidia/Qwen3.8-27B-NVFP4` + `LiquidAI/d1-3B-w8a8` Octop API Benchmark
Measured via `python benchmarks/test_octop_qwen38_d1_benchmark.py` with **15 real safetensors layers (`layers 49..63`, 3,773.04 MB)** of `nvidia/Qwen3.8-27B-NVFP4`, the full **`LiquidAI/d1-3B-w8a8` (3,741.1 MB INT8 W8A8)** decision model, **2.0 GB PagedAttention FP8 KV Cache**, and native **`zeroswap_cuda_ext` (`sm_120`)** at **26.07 GB/s** sustained HtoD bandwidth:

| Octop Delegation Scenario | Initial `d1-3B-w8a8` Action (`output_tokens=0`) | $P(\text{safe})$ Before $\rightarrow$ After | Severity ($0..3$) | External API Secret Leak | KV Cache & `CUDAGraph` Invariant |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Clean Algorithmic Task** (Fenwick Tree) | `delegate_to_external_api` | `0.9797` $\rightarrow$ `0.9797` | `0.03 / 3.0` | **0 Leaks** ✅ | **Bit-Exact (`diff=0.0`)** ✅ |
| **2. Production API Key + Internal IP + PESEL** | `reject_force_local_execution` | `0.0474` $\rightarrow$ `0.2942` *(4 mocked)* | `2.88 / 3.0` | **0 Leaks** ✅ | **Bit-Exact (`diff=0.0`)** ✅ |
| **3. Security Audit / Reverse Engineering** | `sanitize_and_decompose_microtasks` | `0.3775` $\rightarrow$ `0.6225` *(microtasks)* | `1.48 / 3.0` | **0 Leaks** ✅ | **Bit-Exact (`diff=0.0`)** ✅ |
| **4. Raw RSA Private Key + `rm -rf /`** | `reject_force_local_execution` | `0.0100` $\rightarrow$ `0.6792` *(self-corrected)* | `2.95 / 3.0` | **0 Leaks** ✅ | **Bit-Exact (`diff=0.0`)** ✅ |

* **15-Layer `Qwen3.8-27B-NVFP4` (`3,773.04 MB`) Unoverlapped Sync HtoD Restore**: `141.34 ms` (`26.07 GB/s`)
* **15-Layer `Qwen3.8-27B-NVFP4` (`3,773.04 MB`) Pipelined Restore (`Copy || Compute`)**: **`2.02 – 6.76 ms` net TTFT overhead** (>95% hidden underneath resident layers `0..48`)

### 2. Controlled Slab TTFT Latency Hiding Benchmark (`test_ttft_overhead.py`)

| Strategy | Total TTFT (ms) | Net Swap Overhead (ms) | KV Cache Preserved | CUDA Graph Re-Capture Required |
| :--- | :---: | :---: | :---: | :---: |
| **0. Cold Container / Engine Restart** | ~3,500+ ms | +3,400+ ms | ❌ Destroyed | ❌ Full Re-Capture (30–90 s) |
| **1. Baseline (100% Resident in VRAM)** | **87.18 ms** | **0.00 ms** *(ref)* | ✅ Yes | ✅ No |
| **2. Traditional Two-Way Swap (`DtoH + HtoD`)** | 403.22 ms | +316.04 ms | ✅ Yes | ✅ No |
| **3. One-Way Static RAM Mirror (Unoverlapped)** | 105.69 ms | +18.51 ms | ✅ Yes | ✅ No |
| **4. ZeroSwap Pipelined Overwrite (`Copy || Compute`)** | **87.10 ms** | **0.00 ms** | ✅ **100% Intact** | ✅ **Replayed Without Re-Capture** |

---

## Building the Native C++/CUDA Extension (`zeroswap_cuda_ext`)

```bash
cd csrc
export PATH="/usr/local/cuda-13.3/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.3"
export TORCH_CUDA_ARCH_LIST="12.0"
python3 setup_cuda.py build_ext --inplace --force
cp zeroswap_cuda_ext*.so ../
cp zeroswap_cuda_ext*.so ../zeroswap/engine/
```

---

## Quick Start

```bash
# 1. Auto-profile nvidia/Qwen3.8-27B-NVFP4 + LiquidAI/d1-3B-w8a8 and generate plan.json
PYTHONPATH=. python3 -m zeroswap.tune \
  --primary nvidia/Qwen3.8-27B-NVFP4 \
  --secondary LiquidAI/d1-3B-w8a8 \
  --gpu 0 --out plan.json

# 2. Run the full 18-test rigorous pytest suite (including real Qwen3.8 + d1-3B-w8a8 + CUDA ext)
PYTHONPATH=. pytest -v

# 3. Run the 15-layer Qwen3.8-27B-NVFP4 + LiquidAI/d1-3B-w8a8 Octop API GPU benchmark
PYTHONPATH=. python3 benchmarks/test_octop_qwen38_d1_benchmark.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
