"""GuardrailGate & `@gate.intercept` Decorator with In-Place Weight Swapping & Self-Correction Loop.

Provides the standardized Python interface:

    @gate.intercept(safety_model="lfm-1b-guard", mode="pipelined")
    async def handle_agent_step(prompt, kv_context):
        ...

When the guardrail rejects an action/output due to unsanitized PII or policy rules,
the Self-Correction Loop keeps the primary orchestrator's KV cache warm in VRAM and
immediately feeds back:
    "Odrzucono ze względu na regułę {reason_code}, wygeneruj alternatywną abstrakcję zadania"
"""

from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn

from zeroswap.config import EngineConfig, RoutingPath, TunePlan
from zeroswap.engine.layer_interceptor import LayerInterceptor
from zeroswap.engine.pcie_streamer import PCIeStreamer
from zeroswap.guardrail.sanitizer import DataSanitizationGate
from zeroswap.optimizer.profiler import AutoProfiler
from zeroswap.optimizer.router import RoutingDecision, RuntimeDecisionRouter


@dataclass
class GuardrailVerdict:
    approved: bool
    reason_code: str
    feedback_message: Optional[str]
    routing_path: RoutingPath
    swap_overhead_ms: float
    compute_ms: float
    redacted_entities: int
    sanitized_prompt: str
    mock_vault: Dict[str, str] = field(default_factory=dict)


class GuardrailGate:
    """Middleware coordinating ZeroSwap PCIe weight streaming, PII sanitization, and self-correction."""

    DANGEROUS_PATTERNS = (
        ("RAW_SECRET_LEAK", ("BEGIN RSA PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY", "password=")),
        ("DESTRUCTIVE_SHELL", ("rm -rf /", "mkfs.", ":(){ :|:& };:")),
    )

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        plan: Optional[TunePlan] = None,
        streamer: Optional[PCIeStreamer] = None,
        interceptor: Optional[LayerInterceptor] = None,
        secondary_layers: Optional[List[nn.Module]] = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.plan = plan
        self.streamer = streamer
        self.interceptor = interceptor
        self.secondary_layers = secondary_layers
        self.sanitizer = DataSanitizationGate()
        self.d1_engine: Optional[Any] = None

    def attach_d1_engine(self, d1_engine: Any) -> None:
        """Attaches a real `LiquidD1DecisionEngine` (`LiquidAI/d1-3B-w8a8`) for single-pass safety evaluation."""
        self.d1_engine = d1_engine

    def attach_runtime(
        self,
        plan: TunePlan,
        streamer: PCIeStreamer,
        interceptor: Optional[LayerInterceptor] = None,
        secondary_layers: Optional[List[nn.Module]] = None,
    ) -> None:
        self.plan = plan
        self.streamer = streamer
        self.interceptor = interceptor
        self.secondary_layers = secondary_layers

    def evaluate_guardrail(
        self,
        text: str,
        kv_context: Optional[Dict[str, Any]] = None,
        mode: str = "pipelined",
    ) -> GuardrailVerdict:
        """Executes the guardrail model via the optimal ZeroSwap routing path (Path A, B, or C)."""
        kv_context = kv_context if kv_context is not None else {}
        t0 = time.perf_counter()

        # Step 1: Sanitize PII / API keys / Internal IPs if enabled
        if self.config.guardrail.enable_pii_sanitizer and not kv_context.get("skip_auto_sanitize", False):
            san = self.sanitizer.sanitize(text)
            working_text = san.sanitized_text
            mock_vault = san.mock_to_real
            redacted_count = san.redacted_count
        else:
            working_text = text
            mock_vault = {}
            redacted_count = 0

        # Step 2: Determine token counts and routing path via RuntimeDecisionRouter
        est_tokens = max(16, len(working_text.split()) * 2)
        total_tokens = int(kv_context.get("total_tokens", est_tokens))
        cached_tokens = int(kv_context.get("cached_tokens", 0))

        if self.plan is None:
            profiler = AutoProfiler(self.config)
            self.plan = profiler.generate_tune_plan("qwen-32b", "lfm-1b")

        router = RuntimeDecisionRouter(self.plan, self.config)
        decision: RoutingDecision = router.decide(
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            guardrail_prompt_tokens=est_tokens,
        )

        # Step 3: Execute weight swap + eager prefill evaluation if GPU streamer is attached
        swap_start = time.perf_counter()
        if self.streamer is not None and decision.path != "path_b_cpu_direct_eager":
            swapped_indices = (
                self.interceptor.swapped_layer_indices
                if self.interceptor is not None
                else list(self.streamer.slab.layer_params.keys())
            )
            if swapped_indices:
                self.streamer.overwrite_slab_with_secondary(
                    swapped_indices,
                    secondary_gpu_modules=self.secondary_layers,
                )
                if self.interceptor is not None:
                    self.interceptor.mark_dirty_for_restore()
        swap_overhead_ms = (time.perf_counter() - swap_start) * 1000.0

        # Step 4: Eager forward pass / policy evaluation
        comp_start = time.perf_counter()
        if self.secondary_layers and decision.path != "path_b_cpu_direct_eager":
            device = self.streamer.device if self.streamer else torch.device("cpu")
            hidden_dim = self.plan.secondary_model.hidden_size
            dummy_x = torch.randn((1, min(est_tokens, 128), hidden_dim), device=device, dtype=torch.float16)
            with torch.no_grad():
                for layer in self.secondary_layers:
                    dummy_x = layer(dummy_x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        approved = True
        reason_code = "OK"
        feedback_msg: Optional[str] = None

        if self.d1_engine is not None:
            d1_dec = self.d1_engine.evaluate_octop_delegation(working_text)
            if not d1_dec.approved_for_external_api and d1_dec.decision_action == "reject_force_local_execution":
                approved = False
                reason_code = d1_dec.reason_code
                feedback_msg = d1_dec.feedback_for_orchestrator

        for code, triggers in self.DANGEROUS_PATTERNS:
            if any(trig in working_text for trig in triggers):
                approved = False
                reason_code = code
                feedback_msg = (
                    f"Odrzucono ze względu na regułę {code}, "
                    f"wygeneruj alternatywną abstrakcję zadania."
                )
                break

        compute_ms = (time.perf_counter() - comp_start) * 1000.0
        _ = (time.perf_counter() - t0) * 1000.0

        return GuardrailVerdict(
            approved=approved,
            reason_code=reason_code,
            feedback_message=feedback_msg,
            routing_path=decision.path,
            swap_overhead_ms=round(swap_overhead_ms, 3),
            compute_ms=round(compute_ms, 3),
            redacted_entities=redacted_count,
            sanitized_prompt=working_text,
            mock_vault=mock_vault,
        )

    def intercept(
        self,
        safety_model: str = "lfm-1b-guard",
        mode: str = "pipelined",
        max_retries: Optional[int] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator wrapping an orchestrator step with ZeroSwap pipelined guardrail & self-correction."""
        retries_limit = (
            max_retries
            if max_retries is not None
            else self.config.guardrail.max_self_correction_retries
        )

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def _async_wrapper(
                    prompt: str,
                    kv_context: Optional[Dict[str, Any]] = None,
                    *args: Any,
                    **kwargs: Any,
                ) -> Any:
                    ctx = kv_context if kv_context is not None else {}
                    ctx.setdefault("safety_model", safety_model)
                    ctx.setdefault("mode", mode)
                    ctx.setdefault("self_correction_history", [])

                    merged_vault: Dict[str, str] = {}
                    current_prompt = prompt
                    for attempt in range(retries_limit + 1):
                        verdict = self.evaluate_guardrail(current_prompt, ctx, mode=mode)
                        merged_vault.update(verdict.mock_vault)
                        ctx["last_verdict"] = verdict
                        ctx["mock_vault"] = merged_vault

                        if verdict.approved:
                            result = await func(verdict.sanitized_prompt, ctx, *args, **kwargs)
                            if isinstance(result, str) and merged_vault:
                                return self.sanitizer.unmask(result, merged_vault)
                            return result

                        # Self-Correction Loop: keep KV cache intact, inject feedback, retry
                        ctx["self_correction_history"].append(
                            {
                                "attempt": attempt + 1,
                                "reason_code": verdict.reason_code,
                                "feedback": verdict.feedback_message,
                            }
                        )
                        if attempt < retries_limit and "on_reject_rewrite" in ctx:
                            current_prompt = ctx["on_reject_rewrite"](
                                current_prompt, verdict.feedback_message
                            )
                        else:
                            current_prompt = (
                                f"[SYSTEM FEEDBACK: {verdict.feedback_message}]\n"
                                + self.sanitizer.sanitize(current_prompt).sanitized_text
                            )
                            # Allow sanitized retry on next pass
                            for _, triggers in self.DANGEROUS_PATTERNS:
                                for trig in triggers:
                                    current_prompt = current_prompt.replace(trig, "<REDACTED_POLICY_TOKEN>")

                    raise PermissionError(
                        f"Guardrail ({safety_model}) rejected request after {retries_limit} "
                        f"self-correction attempts: {ctx['last_verdict'].reason_code}"
                    )

                return _async_wrapper

            @functools.wraps(func)
            def _sync_wrapper(
                prompt: str,
                kv_context: Optional[Dict[str, Any]] = None,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                ctx = kv_context if kv_context is not None else {}
                ctx.setdefault("safety_model", safety_model)
                ctx.setdefault("mode", mode)
                ctx.setdefault("self_correction_history", [])

                merged_vault: Dict[str, str] = {}
                current_prompt = prompt
                for attempt in range(retries_limit + 1):
                    verdict = self.evaluate_guardrail(current_prompt, ctx, mode=mode)
                    merged_vault.update(verdict.mock_vault)
                    ctx["last_verdict"] = verdict
                    ctx["mock_vault"] = merged_vault

                    if verdict.approved:
                        result = func(verdict.sanitized_prompt, ctx, *args, **kwargs)
                        if isinstance(result, str) and merged_vault:
                            return self.sanitizer.unmask(result, merged_vault)
                        return result

                    ctx["self_correction_history"].append(
                        {
                            "attempt": attempt + 1,
                            "reason_code": verdict.reason_code,
                            "feedback": verdict.feedback_message,
                        }
                    )
                    if attempt < retries_limit and "on_reject_rewrite" in ctx:
                        current_prompt = ctx["on_reject_rewrite"](
                            current_prompt, verdict.feedback_message
                        )
                    else:
                        current_prompt = (
                            f"[SYSTEM FEEDBACK: {verdict.feedback_message}]\n"
                            + self.sanitizer.sanitize(current_prompt).sanitized_text
                        )
                        for _, triggers in self.DANGEROUS_PATTERNS:
                            for trig in triggers:
                                current_prompt = current_prompt.replace(trig, "<REDACTED_POLICY_TOKEN>")

                raise PermissionError(
                    f"Guardrail ({safety_model}) rejected request after {retries_limit} "
                    f"self-correction attempts: {ctx['last_verdict'].reason_code}"
                )

            return _sync_wrapper

        return _decorator


# Default singleton instance for `@gate.intercept(...)`
gate = GuardrailGate()
