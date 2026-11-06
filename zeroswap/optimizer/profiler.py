"""AutoProfiler: Zero-Config Hardware & Topology Profiler for ZeroSwap.

Executes a 3-stage startup probe:
1. `probe_pcie_bandwidth`: Measures real-world Pinned-Host-to-Device DMA bandwidth via CUDA events.
2. `analyze_model_topology`: Parses model `config.json` & `model.safetensors.index.json` (or architectural
   presets) to compute exact per-layer weight footprint (MB) and per-token KV cache footprint (bytes).
3. `probe_layer_compute`: Measures single-layer forward pass compute latency (`T_compute`) on GPU
   and solves the Break-Even Delta Token Threshold (`Delta N_crit`) and Asymmetry Strategy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from zeroswap.config import EngineConfig, ModelTopology, StrategyVariant, TunePlan
from zeroswap.optimizer.kv_budget import KVBudgetPlanner


MODEL_PRESETS: Dict[str, Dict[str, float | int]] = {
    "qwen3.8-27b-nvfp4": {
        "num_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_kv_heads": 4,
        "head_dim": 256,
        "bytes_per_param": 0.58,  # NVFP4 MLP + FP8 Attention (~251.72 MB/layer)
        "total_weight_mb": 20905.9,
        "layer_size_mb": 251.72,
    },
    "nvidia/qwen3.8-27b-nvfp4": {
        "num_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_kv_heads": 4,
        "head_dim": 256,
        "bytes_per_param": 0.58,
        "total_weight_mb": 20905.9,
        "layer_size_mb": 251.72,
    },
    "liquidai/d1-3b-w8a8": {
        "num_layers": 30,
        "hidden_size": 2048,
        "intermediate_size": 10752,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "bytes_per_param": 1.0,  # INT8 W8A8 torchao (~3,741.1 MB total)
        "total_weight_mb": 3741.1,
        "layer_size_mb": 124.7,
    },
    "d1-3b-w8a8": {
        "num_layers": 30,
        "hidden_size": 2048,
        "intermediate_size": 10752,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "bytes_per_param": 1.0,
        "total_weight_mb": 3741.1,
        "layer_size_mb": 124.7,
    },
    "qwen-32b": {
        "num_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 27648,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "bytes_per_param": 0.55,
        "total_weight_mb": 19840.0,
    },
    "nemotron-30b": {
        "num_layers": 56,
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "bytes_per_param": 0.55,
        "total_weight_mb": 17920.0,
    },
    "qwen-14b": {
        "num_layers": 48,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "bytes_per_param": 0.55,
        "total_weight_mb": 9200.0,
    },
    "llama-8b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "bytes_per_param": 0.55,
        "total_weight_mb": 4800.0,
    },
    "llama-3b": {
        "num_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_kv_heads": 8,
        "bytes_per_param": 1.0,
        "total_weight_mb": 3200.0,
    },
    "lfm-1b": {
        "num_layers": 24,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_attention_heads": 16,
        "num_kv_heads": 8,
        "bytes_per_param": 1.0,
        "total_weight_mb": 1800.0,
    },
    "qwen3-0.6b": {
        "num_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_kv_heads": 8,
        "bytes_per_param": 2.0,
        "total_weight_mb": 1200.0,
    },
}


class AutoProfiler:
    """Zero-config profiler for PCIe DMA bandwidth, model topology, and compute/transfer overlap."""

    def __init__(self, config: Optional[EngineConfig] = None, gpu_index: int = 0) -> None:
        self.config = config or EngineConfig()
        self.gpu_index = gpu_index
        self.has_cuda = torch.cuda.is_available() and gpu_index < torch.cuda.device_count()
        self.device = torch.device(f"cuda:{gpu_index}" if self.has_cuda else "cpu")

    def get_gpu_info(self) -> Tuple[str, float]:
        if self.has_cuda:
            props = torch.cuda.get_device_properties(self.gpu_index)
            total_mb = props.total_memory / (1024.0 * 1024.0)
            return props.name, round(total_mb, 2)
        return "Simulated GPU (32 GB)", 32768.0

    def probe_pcie_bandwidth(self, buffer_mb: int = 256, num_iters: int = 5) -> Tuple[float, str]:
        """Measures real asynchronous Pinned-RAM-to-VRAM HtoD bandwidth in GB/s."""
        if not self.has_cuda:
            return 24.8, "PCIe 4.0 x16 (Simulated)"

        num_bytes = buffer_mb * 1024 * 1024
        host_buf = torch.empty(num_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        host_buf.fill_(42)
        dev_buf = torch.empty(num_bytes, dtype=torch.uint8, device=self.device)
        stream = torch.cuda.Stream(device=self.device, priority=-1)

        # Warm-up
        with torch.cuda.stream(stream):
            dev_buf.copy_(host_buf, non_blocking=True)
        stream.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream):
            start_ev.record(stream)
            for _ in range(num_iters):
                dev_buf.copy_(host_buf, non_blocking=True)
            end_ev.record(stream)
        end_ev.synchronize()

        elapsed_ms = max(0.001, start_ev.elapsed_time(end_ev))
        total_gb = (num_bytes * num_iters) / (1024.0**3)
        bw_gbps = total_gb / (elapsed_ms / 1000.0)

        if bw_gbps >= 36.0:
            gen = "PCIe 5.0 x16"
        elif bw_gbps >= 18.0:
            gen = "PCIe 4.0 x16"
        else:
            gen = "PCIe 3.0 x16 / Split Lane"

        del host_buf, dev_buf
        return round(bw_gbps, 2), gen

    def analyze_model_topology(self, model_id_or_path: str) -> ModelTopology:
        """Reads `config.json` from disk or resolves architectural preset parameters."""
        kv_dtype_bytes = 1.0 if self.config.context_priority.kv_cache_dtype.lower() in ("fp8", "int8", "fp8_e4m3") else 2.0
        key = model_id_or_path.lower().strip()

        # Check if path with config.json exists
        cand_path = Path(model_id_or_path)
        cfg_file = cand_path / "config.json" if cand_path.is_dir() else cand_path
        if cfg_file.is_file() and cfg_file.name == "config.json":
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
            text_cfg = raw.get("text_config", raw)
            num_layers = int(text_cfg.get("num_hidden_layers", 32))
            hidden_size = int(text_cfg.get("hidden_size", 4096))
            intermediate_size = int(text_cfg.get("intermediate_size", hidden_size * 4))
            num_heads = int(text_cfg.get("num_attention_heads", 32))
            num_kv_heads = int(text_cfg.get("num_key_value_heads", num_heads))
            head_dim = int(text_cfg.get("head_dim") or (hidden_size // max(1, num_heads)))

            quant_cfg = raw.get("quantization_config", {})
            quant_method = str(quant_cfg.get("quant_method", "")).lower()
            if quant_method == "torchao":
                bytes_per_param = 1.0
            elif quant_cfg:
                bytes_per_param = 0.58
            else:
                bytes_per_param = 2.0

            model_dir = cfg_file.parent
            idx_file = model_dir / "model.safetensors.index.json"
            single_sf = model_dir / "model.safetensors"
            if idx_file.is_file():
                idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
                total_bytes = float(idx_data.get("metadata", {}).get("total_size", 0))
                if total_bytes > 0:
                    total_mb = total_bytes / (1024.0 * 1024.0)
                    # Decoder layers represent ~77% of hybrid VLM/MTP checkpoints or 95% of pure text
                    has_vision = "vision_config" in raw
                    layer_fraction = 0.771 if has_vision else 0.94
                    layer_mb = (total_mb * layer_fraction) / max(1, num_layers)
                else:
                    params_per_layer = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
                    layer_mb = (params_per_layer * bytes_per_param) / (1024.0 * 1024.0)
                    total_mb = layer_mb * num_layers
            elif single_sf.is_file():
                total_mb = single_sf.stat().st_size / (1024.0 * 1024.0)
                layer_mb = total_mb / max(1, num_layers)
            else:
                params_per_layer = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
                layer_mb = (params_per_layer * bytes_per_param) / (1024.0 * 1024.0)
                total_mb = layer_mb * num_layers

            kv_bytes_per_token = 2.0 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes

            return ModelTopology(
                model_id=model_id_or_path,
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_heads,
                num_kv_heads=num_kv_heads,
                bytes_per_param=bytes_per_param,
                total_weight_mb=round(total_mb, 2),
                layer_size_mb=round(layer_mb, 2),
                kv_bytes_per_token=round(kv_bytes_per_token, 2),
            )

        # Match preset or heuristic size from model name
        preset = MODEL_PRESETS.get(key)
        if preset is None:
            for preset_name, p_val in MODEL_PRESETS.items():
                if preset_name in key:
                    preset = p_val
                    break
        if preset is None:
            if "27b" in key or "qwen3.8" in key:
                preset = MODEL_PRESETS["qwen3.8-27b-nvfp4"]
            elif "d1-3b" in key:
                preset = MODEL_PRESETS["liquidai/d1-3b-w8a8"]
            elif "32b" in key or "30b" in key:
                preset = MODEL_PRESETS["qwen-32b"]
            else:
                preset = MODEL_PRESETS["lfm-1b"]

        num_layers = int(preset["num_layers"])
        hidden_size = int(preset["hidden_size"])
        intermediate_size = int(preset["intermediate_size"])
        num_heads = int(preset["num_attention_heads"])
        num_kv_heads = int(preset["num_kv_heads"])
        bytes_per_param = float(preset["bytes_per_param"])
        total_mb = float(preset["total_weight_mb"])
        layer_mb = float(preset.get("layer_size_mb", total_mb / max(1, num_layers)))
        head_dim = int(preset.get("head_dim", hidden_size // max(1, num_heads)))
        kv_bytes_per_token = 2.0 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes

        return ModelTopology(
            model_id=model_id_or_path,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bytes_per_param=bytes_per_param,
            total_weight_mb=round(total_mb, 2),
            layer_size_mb=round(layer_mb, 2),
            kv_bytes_per_token=round(kv_bytes_per_token, 2),
        )

    def probe_layer_compute_ms(
        self,
        topology: ModelTopology,
        seq_len: int = 1024,
        num_iters: int = 5,
    ) -> Tuple[float, float]:
        """Measures single-layer GPU forward pass time (ms per 1k tokens) and total model prefill throughput."""
        if not self.has_cuda:
            layer_ms = 1.56
            throughput = 1000.0 / ((layer_ms * topology.num_layers) / 1000.0)
            return round(layer_ms, 3), round(throughput, 1)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        d_model = topology.hidden_size
        d_ffn = min(topology.intermediate_size, d_model * 4)

        qkv = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)
        gate_up = nn.Linear(d_model, d_ffn, bias=False, device=self.device, dtype=dtype)
        down = nn.Linear(d_ffn, d_model, bias=False, device=self.device, dtype=dtype)
        x = torch.randn((1, seq_len, d_model), device=self.device, dtype=dtype)

        with torch.no_grad():
            for _ in range(2):
                h = qkv(x)
                h = down(torch.nn.functional.silu(gate_up(h)))
            torch.cuda.synchronize(self.device)

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(num_iters):
                h = qkv(x)
                h = down(torch.nn.functional.silu(gate_up(h)))
            end_ev.record()
            end_ev.synchronize()

        single_layer_ms = max(0.05, start_ev.elapsed_time(end_ev) / num_iters)
        scale_factor = topology.intermediate_size / max(1, d_ffn)
        scaled_layer_ms = single_layer_ms * max(1.0, scale_factor * 0.65)
        full_model_ms_per_1k = scaled_layer_ms * topology.num_layers
        tokens_per_sec = seq_len / (full_model_ms_per_1k / 1000.0)

        del qkv, gate_up, down, x
        return round(scaled_layer_ms, 3), round(tokens_per_sec, 1)

    def generate_tune_plan(self, primary_id: str, secondary_id: str) -> TunePlan:
        """Runs the full Zero-Config Auto-Tuning benchmark and computes the optimal execution plan."""
        gpu_name, total_vram_mb = self.get_gpu_info()
        pcie_bw_gbps, pcie_gen = self.probe_pcie_bandwidth()

        primary_topo = self.analyze_model_topology(primary_id)
        secondary_topo = self.analyze_model_topology(secondary_id)

        # Asymmetry Strategy Detection:
        # Variant A (Standard): Primary Orchestrator >= Secondary Guardrail -> Partial Layer Swap
        # Variant B (Inverted): Primary Orchestrator < Secondary Analyst -> Full Role Pivot
        if primary_topo.total_weight_mb >= secondary_topo.total_weight_mb:
            strategy: StrategyVariant = "partial_layer_swap"
            base_topo = primary_topo
            guest_topo = secondary_topo
        else:
            strategy = "full_role_pivot"
            base_topo = secondary_topo
            guest_topo = primary_topo

        kv_planner = KVBudgetPlanner(self.config)
        budget = kv_planner.plan_budget(
            total_vram_mb=total_vram_mb,
            base_model=base_topo,
            guest_model=guest_topo,
        )

        layer_compute_ms, prefill_tps = self.probe_layer_compute_ms(base_topo)
        pcie_bw_mbps = pcie_bw_gbps * 1024.0

        transfer_slab_ms = (budget.dynamic_slab_mb / max(1.0, pcie_bw_mbps)) * 1000.0
        transfer_single_layer_ms = (base_topo.layer_size_mb / max(1.0, pcie_bw_mbps)) * 1000.0

        # Solve Break-Even Delta Token Threshold (Delta N_crit):
        # Delta N_crit = (Swapped Weight Size (MB) / PCIe Bandwidth (MB/s)) * Prefill Throughput (tokens/s)
        resident_ratio = max(0.1, budget.resident_layers / max(1, base_topo.num_layers))
        effective_resident_tps = prefill_tps / resident_ratio
        break_even_delta_tokens = int(
            math.ceil((transfer_slab_ms / 1000.0) * prefill_tps / resident_ratio)
        )

        # For default prefill >= break_even_delta_tokens, transfer is 100% hidden under resident layers
        resident_compute_ms_at_2k = (2048.0 / max(1.0, effective_resident_tps)) * 1000.0
        net_ttft_overhead_ms = max(0.0, transfer_slab_ms - resident_compute_ms_at_2k)

        return TunePlan(
            gpu_index=self.gpu_index,
            gpu_name=gpu_name,
            total_vram_mb=total_vram_mb,
            pcie_bandwidth_gbps=pcie_bw_gbps,
            pcie_generation_estimate=pcie_gen,
            primary_model=primary_topo,
            secondary_model=secondary_topo,
            strategy_variant=strategy,
            resident_layers=budget.resident_layers,
            swapped_layers=budget.swapped_layers,
            dynamic_slab_mb=budget.dynamic_slab_mb,
            ping_pong_buffer_mb=budget.ping_pong_buffer_mb,
            kv_cache_pool_mb=budget.kv_cache_pool_mb,
            kv_cache_max_tokens=budget.kv_cache_max_tokens,
            kv_cache_blocks=budget.kv_cache_blocks,
            single_layer_compute_ms_per_1k_tokens=layer_compute_ms,
            prefill_throughput_tokens_per_sec=prefill_tps,
            transfer_time_swapped_slab_ms=round(transfer_slab_ms, 2),
            transfer_time_single_layer_ms=round(transfer_single_layer_ms, 2),
            break_even_delta_tokens=break_even_delta_tokens,
            degraded_quantization_fallback=budget.degraded_quantization_fallback,
            estimated_net_ttft_overhead_ms=round(net_ttft_overhead_ms, 2),
        )
