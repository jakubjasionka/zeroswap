"""OpenAI-Compatible vLLM Middleware & Local ZeroSwap Control-Plane Server.

Provides:
- `POST /v1/chat/completions`:
    1. Intercepts incoming agent/user prompts via `GuardrailGate` & `DataSanitizationGate`.
    2. Masks PII, API keys, and internal IPs with reversible synthetic tokens.
    3. Selects the optimal ZeroSwap execution path (`Path A`, `Path B`, or `Path C`) using
       `RuntimeDecisionRouter` based on prefix cache hit rate and token delta `Delta N`.
    4. Forwards the sanitized prompt to the upstream vLLM server (or executes locally),
       performs self-correction retries if rejected by the guardrail, and unmasks synthetic
       tokens in the returned response.
- `GET /v1/zeroswap/status`: Returns active `TunePlan`, KV cache headroom, and PCIe metrics.
- `POST /v1/zeroswap/tune`: Runs on-demand hardware auto-profiling (`AutoProfiler`).
- `POST /v1/zeroswap/verify`: Runs a standalone guardrail + PII sanitization pass.
"""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Any, Dict, List, Optional
import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import uvicorn

from zeroswap.config import EngineConfig, TunePlan
from zeroswap.guardrail.gate import GuardrailGate
from zeroswap.optimizer.profiler import AutoProfiler
from zeroswap.optimizer.router import RuntimeDecisionRouter


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen-32b"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    cached_tokens: int = Field(default=0, description="Prefix-cached token count from RadixAttention")


class TuneRequest(BaseModel):
    primary_model: str = "qwen-32b"
    secondary_model: str = "lfm-1b"
    gpu_index: int = 0
    max_context_length: Optional[int] = None


class VerifyRequest(BaseModel):
    prompt: str
    total_tokens: int = 2048
    cached_tokens: int = 0


def create_zeroswap_app(
    upstream_vllm_url: Optional[str] = None,
    primary_model: str = "qwen-32b",
    secondary_model: str = "lfm-1b",
    gpu_index: int = 0,
) -> FastAPI:
    app = FastAPI(
        title="ZeroSwap vLLM Control-Plane & Guardrail Interceptor",
        version="0.1.0",
        description="Zero-Latency In-Flight Weight Swapping & PII Sanitization Middleware for vLLM",
    )

    cfg = EngineConfig()
    profiler = AutoProfiler(config=cfg, gpu_index=gpu_index)
    state: Dict[str, Any] = {
        "config": cfg,
        "profiler": profiler,
        "plan": profiler.generate_tune_plan(primary_model, secondary_model),
        "upstream_vllm_url": upstream_vllm_url,
        "requests_processed": 0,
        "self_corrections_triggered": 0,
        "total_entities_redacted": 0,
    }
    gate_inst = GuardrailGate(config=cfg, plan=state["plan"])

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "engine": "zeroswap-vllm"}

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        plan: TunePlan = state["plan"]
        return {
            "object": "list",
            "data": [
                {
                    "id": plan.primary_model.model_id,
                    "object": "model",
                    "owned_by": "zeroswap-vllm",
                    "role": "primary_orchestrator",
                },
                {
                    "id": plan.secondary_model.model_id,
                    "object": "model",
                    "owned_by": "zeroswap-vllm",
                    "role": "secondary_guardrail",
                },
            ],
        }

    @app.get("/v1/zeroswap/status")
    async def zeroswap_status() -> Dict[str, Any]:
        plan: TunePlan = state["plan"]
        return {
            "status": "active",
            "upstream_vllm_url": state["upstream_vllm_url"],
            "telemetry": {
                "requests_processed": state["requests_processed"],
                "self_corrections_triggered": state["self_corrections_triggered"],
                "total_entities_redacted": state["total_entities_redacted"],
            },
            "plan": {
                "gpu_name": plan.gpu_name,
                "pcie_bandwidth_gbps": plan.pcie_bandwidth_gbps,
                "strategy_variant": plan.strategy_variant,
                "resident_layers": plan.resident_layers,
                "swapped_layers": plan.swapped_layers,
                "dynamic_slab_mb": plan.dynamic_slab_mb,
                "kv_cache_max_tokens": plan.kv_cache_max_tokens,
                "break_even_delta_tokens": plan.break_even_delta_tokens,
                "estimated_net_ttft_overhead_ms": plan.estimated_net_ttft_overhead_ms,
            },
        }

    @app.post("/v1/zeroswap/tune")
    async def zeroswap_tune(req: TuneRequest) -> Dict[str, Any]:
        if req.max_context_length is not None:
            cfg.context_priority.max_context_length = req.max_context_length
        prof = AutoProfiler(config=cfg, gpu_index=req.gpu_index)
        new_plan = prof.generate_tune_plan(req.primary_model, req.secondary_model)
        state["plan"] = new_plan
        gate_inst.plan = new_plan
        router = RuntimeDecisionRouter(new_plan, cfg)
        return {
            "plan": new_plan,
            "sample_routing": {
                "cold_20k": router.decide(20000, 0),
                "high_cache_hit_micro": router.decide(64000, 63750, 250),
            },
        }

    @app.post("/v1/zeroswap/verify")
    async def zeroswap_verify(req: VerifyRequest) -> Dict[str, Any]:
        verdict = gate_inst.evaluate_guardrail(
            text=req.prompt,
            kv_context={"total_tokens": req.total_tokens, "cached_tokens": req.cached_tokens},
        )
        return {
            "approved": verdict.approved,
            "reason_code": verdict.reason_code,
            "feedback_message": verdict.feedback_message,
            "routing_path": verdict.routing_path,
            "swap_overhead_ms": verdict.swap_overhead_ms,
            "compute_ms": verdict.compute_ms,
            "redacted_entities": verdict.redacted_entities,
            "sanitized_prompt": verdict.sanitized_prompt,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, response: Response) -> Dict[str, Any]:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        combined_prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        est_total_tokens = max(32, len(combined_prompt.split()) * 2 + req.cached_tokens)
        kv_ctx: Dict[str, Any] = {
            "total_tokens": est_total_tokens,
            "cached_tokens": req.cached_tokens,
        }

        @gate_inst.intercept(safety_model=state["plan"].secondary_model.model_id, mode="pipelined")
        async def _execute_step(sanitized_prompt: str, ctx: Dict[str, Any]) -> str:
            if state["upstream_vllm_url"]:
                sanitized_messages = [
                    {
                        "role": m.role,
                        "content": gate_inst.sanitizer.sanitize(m.content).sanitized_text,
                    }
                    for m in req.messages
                ]
                if ctx.get("self_correction_history"):
                    last_fb = ctx["self_correction_history"][-1]["feedback"]
                    sanitized_messages.insert(
                        0, {"role": "system", "content": f"[GUARDRAIL FEEDBACK]: {last_fb}"}
                    )
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(
                        f"{state['upstream_vllm_url'].rstrip('/')}/v1/chat/completions",
                        json={
                            "model": req.model,
                            "messages": sanitized_messages,
                            "max_tokens": req.max_tokens,
                            "temperature": req.temperature,
                        },
                    )
                    r.raise_for_status()
                    data = r.json()
                    return data["choices"][0]["message"]["content"]
            return (
                f"[ZeroSwap Local Control Plane | Verified by {state['plan'].secondary_model.model_id}] "
                f"Processed sanitized payload: {sanitized_prompt}"
            )

        output_text = await _execute_step(combined_prompt, kv_ctx)
        verdict = kv_ctx["last_verdict"]

        state["requests_processed"] += 1
        state["total_entities_redacted"] += verdict.redacted_entities
        state["self_corrections_triggered"] += len(kv_ctx.get("self_correction_history", []))

        response.headers["X-ZeroSwap-Routing-Path"] = verdict.routing_path
        response.headers["X-ZeroSwap-Swap-Overhead-Ms"] = str(verdict.swap_overhead_ms)
        response.headers["X-ZeroSwap-Redacted-Entities"] = str(verdict.redacted_entities)
        response.headers["X-ZeroSwap-KV-Invariant"] = "preserved"

        return {
            "id": f"chatcmpl-zeroswap-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": est_total_tokens,
                "completion_tokens": max(1, len(output_text.split())),
                "total_tokens": est_total_tokens + max(1, len(output_text.split())),
            },
            "zeroswap_metadata": {
                "routing_path": verdict.routing_path,
                "swap_overhead_ms": verdict.swap_overhead_ms,
                "guardrail_compute_ms": verdict.compute_ms,
                "redacted_entities": verdict.redacted_entities,
                "self_correction_retries": len(kv_ctx.get("self_correction_history", [])),
                "kv_cache_invariant": True,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ZeroSwap OpenAI-Compatible vLLM Interceptor Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream-vllm", type=str, default=None, help="Upstream vLLM URL (e.g. http://localhost:8000)")
    parser.add_argument("--primary", type=str, default="qwen-32b")
    parser.add_argument("--secondary", type=str, default="lfm-1b")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    app = create_zeroswap_app(
        upstream_vllm_url=args.upstream_vllm,
        primary_model=args.primary,
        secondary_model=args.secondary,
        gpu_index=args.gpu,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
