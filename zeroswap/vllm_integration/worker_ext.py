"""ZeroSwapWorkerExtension: Native vLLM Worker Extension for In-Flight Layer Swapping.

Usage with `vllm serve` or `vllm.LLM`:
    vllm serve <primary_model> \\
        --worker-extension-cls zeroswap.vllm_integration.ZeroSwapWorkerExtension

Exposes RPC methods callable via `engine.collective_rpc("zeroswap_...")`:
- `zeroswap_init(num_swapped_layers=3, pin_memory=True)`:
    Locates the decoder `nn.ModuleList` inside `self.model_runner.model`, registers the tail
    layers in `StaticSlab`, mirrors their Golden Image into `PinnedHostPool` (page-locked RAM),
    registers the PagedAttention `kv_caches` to enforce the KV Cache Invariant, and attaches
    `LayerInterceptor` forward hooks without invalidating captured CUDA Graphs.
- `zeroswap_run_guardrail_step(prompt_text, total_tokens, cached_tokens)`:
    Overwrites the tail slab in-place (0 ms DtoH), runs the guardrail check, marks the
    interceptor so the very next orchestrator forward pass asynchronously restores the tail
    layers in parallel with layers `0 .. K-1` execution, and verifies KV Cache invariance.
- `zeroswap_verify_invariants()`:
    Confirms that all CUDA Graph parameter pointers (`data_ptr()`) and PagedAttention KV cache
    pools remain 100% untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from zeroswap.engine.layer_interceptor import LayerInterceptor
from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.engine.pinned_host import PinnedHostPool
from zeroswap.engine.static_slab import StaticSlab
from zeroswap.guardrail.gate import GuardrailGate


class ZeroSwapWorkerExtension:
    """Mixin class injected into vLLM's GPU `Worker` via `--worker-extension-cls`."""

    _zeroswap_slab: Optional[StaticSlab] = None
    _zeroswap_host_pool: Optional[PinnedHostPool] = None
    _zeroswap_streamer: Optional[PCIeStreamer] = None
    _zeroswap_interceptor: Optional[LayerInterceptor] = None
    _zeroswap_gate: Optional[GuardrailGate] = None
    _zeroswap_swapped_indices: List[int] = []

    def _locate_decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        """Finds the transformer decoder `nn.ModuleList` across vLLM model architectures."""
        for attr_path in (
            ("model", "layers"),
            ("language_model", "model", "layers"),
            ("transformer", "h"),
            ("layers",),
        ):
            cur: Any = model
            found = True
            for part in attr_path:
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    found = False
                    break
            if found and isinstance(cur, (nn.ModuleList, list)) and len(cur) > 0:
                return cur  # type: ignore[return-value]

        for _, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 4:
                return module
        raise RuntimeError("Could not locate decoder ModuleList inside vLLM model.")

    def _locate_kv_caches(self) -> Optional[List[torch.Tensor] | Dict[str, torch.Tensor]]:
        runner = getattr(self, "model_runner", None)
        if runner is not None:
            for attr in ("kv_caches", "gpu_cache"):
                val = getattr(runner, attr, None)
                if val:
                    return val
        for attr in ("kv_cache", "gpu_cache"):
            val = getattr(self, attr, None)
            if val:
                return val
        return None

    def zeroswap_init(
        self,
        num_swapped_layers: int = 3,
        pin_memory: bool = True,
    ) -> Dict[str, Any]:
        """Initializes StaticSlab, PinnedHostPool, PCIeStreamer, and LayerInterceptor on the vLLM worker."""
        model = self.get_model() if hasattr(self, "get_model") else self.model_runner.model
        layers = self._locate_decoder_layers(model)
        total_layers = len(layers)
        num_swapped = max(1, min(total_layers - 1, int(num_swapped_layers)))
        swapped_indices = list(range(total_layers - num_swapped, total_layers))

        device = next(model.parameters()).device
        slab = StaticSlab(device=device)
        host_pool = PinnedHostPool(pin_memory=pin_memory)

        total_slab_bytes = 0
        for idx in swapped_indices:
            total_slab_bytes += slab.register_swappable_layer(idx, layers[idx])
            host_pool.mirror_primary_layer(idx, layers[idx])

        # Register secondary guardrail weights in pinned RAM (synthetic or loaded module)
        host_pool.mirror_secondary_model([layers[idx] for idx in swapped_indices])

        kv_caches = self._locate_kv_caches()
        if kv_caches is not None:
            slab.register_kv_cache_pool(kv_caches)

        streamer = PCIeStreamer(slab=slab, host_pool=host_pool, device=device)
        interceptor = LayerInterceptor(
            layers=layers,
            swapped_layer_indices=swapped_indices,
            slab=slab,
            streamer=streamer,
        )
        interceptor.attach()

        gate_inst = GuardrailGate(streamer=streamer, interceptor=interceptor)

        self._zeroswap_slab = slab
        self._zeroswap_host_pool = host_pool
        self._zeroswap_streamer = streamer
        self._zeroswap_interceptor = interceptor
        self._zeroswap_gate = gate_inst
        self._zeroswap_swapped_indices = swapped_indices

        return {
            "status": "initialized",
            "device": str(device),
            "total_layers": total_layers,
            "resident_layers": total_layers - num_swapped,
            "swapped_layer_indices": swapped_indices,
            "slab_megabytes": round(total_slab_bytes / (1024.0 * 1024.0), 2),
            "pinned_host_megabytes": round(host_pool.total_pinned_bytes / (1024.0 * 1024.0), 2),
            "kv_cache_registered": kv_caches is not None,
        }

    def zeroswap_run_guardrail_step(
        self,
        prompt_text: str,
        total_tokens: int = 2048,
        cached_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Executes in-flight guardrail check on the vLLM worker and arms async layer restoration."""
        if self._zeroswap_gate is None or self._zeroswap_slab is None:
            raise RuntimeError("ZeroSwapWorkerExtension not initialized. Call zeroswap_init first.")

        verdict = self._zeroswap_gate.evaluate_guardrail(
            text=prompt_text,
            kv_context={"total_tokens": total_tokens, "cached_tokens": cached_tokens},
        )
        invariant_ok = self.zeroswap_verify_invariants()["invariant_preserved"]
        return {
            "approved": verdict.approved,
            "reason_code": verdict.reason_code,
            "feedback_message": verdict.feedback_message,
            "routing_path": verdict.routing_path,
            "swap_overhead_ms": verdict.swap_overhead_ms,
            "compute_ms": verdict.compute_ms,
            "redacted_entities": verdict.redacted_entities,
            "sanitized_prompt": verdict.sanitized_prompt,
            "kv_and_cudagraph_invariant_preserved": invariant_ok,
        }

    def zeroswap_verify_invariants(self) -> Dict[str, Any]:
        """Checks that all swappable layer pointers and PagedAttention KV caches are untouched."""
        if self._zeroswap_slab is None:
            return {"invariant_preserved": False, "error": "Not initialized"}
        for idx in self._zeroswap_swapped_indices:
            self._zeroswap_slab.verify_layer_pointers_invariant(idx)
        kv_caches = self._locate_kv_caches()
        if kv_caches is not None and self._zeroswap_slab.kv_cache_signatures:
            self._zeroswap_slab.verify_kv_cache_invariant(kv_caches)
        return {
            "invariant_preserved": True,
            "slab_state": self._zeroswap_slab.slab_state,
            "swapped_layer_indices": self._zeroswap_swapped_indices,
        }
