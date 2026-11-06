"""LiquidD1DecisionEngine: Real `LiquidAI/d1-3B-w8a8` System-One Decision Model for Octop API Safety.

Evaluates candidate subtasks/prompts in ONE single forward pass with ZERO output tokens (`output_tokens=0`)
before they are delegated to an external commercial Frontier API (Claude Code, Cursor, OpenAI, etc.)
under the strict threat model that the external API:
  1. Permanently retains all submitted prompts/logs (`external_api_retention = True`), and
  2. Uses submitted prompts for future model training (`external_api_training = True`).
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn


DEFAULT_D1_REPO = "LiquidAI/d1-3B-w8a8"


def apply_torchao_d1_compat_shim() -> None:
    """Ensures `LiquidAI/d1-3B-w8a8` (torchao v2 config) loads cleanly on torchao 0.17 and 0.18+."""
    try:
        import torchao.core.config as tcfg
        import torchao.prototype.safetensors.safetensors_utils as tsafe
        import torchao.prototype.safetensors.safetensors_support as ts_sup
        import transformers.utils.quantization_config as hf_qcfg

        if not getattr(tcfg, "_zeroswap_shimmed", False):
            orig_cfd = tcfg.config_from_dict

            def _compat_cfd(data: Any) -> Any:
                if isinstance(data, dict) and isinstance(data.get("_data"), dict):
                    data = dict(data)
                    data["_data"] = {
                        k: v for k, v in data["_data"].items() if k != "reduce_range"
                    }
                return orig_cfd(data)

            tcfg.config_from_dict = _compat_cfd
            hf_qcfg.config_from_dict = _compat_cfd
            tcfg._zeroswap_shimmed = True  # type: ignore[attr-defined]

        if not getattr(tsafe, "_zeroswap_shimmed", False):
            orig_ofd = tsafe.object_from_dict

            def _compat_ofd(data: Any) -> Any:
                if isinstance(data, dict) and isinstance(data.get("_data"), dict):
                    data = dict(data)
                    data["_data"] = {
                        k: v for k, v in data["_data"].items() if k != "reduce_range"
                    }
                return orig_ofd(data)

            tsafe.object_from_dict = _compat_ofd
            ts_sup.object_from_dict = _compat_ofd
            tsafe._zeroswap_shimmed = True  # type: ignore[attr-defined]
    except ImportError:
        pass


OCTOP_EXTERNAL_API_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "safe_for_external_retention": {
        "type": "noul",
        "instructions": (
            "Is this prompt completely safe to send to an external third-party AI API "
            "that permanently retains and trains on all submitted text?"
        ),
        "criteria": {
            "true": (
                "Safe: Contains only generic algorithms, abstract math, or already-masked "
                "synthetic placeholder tokens with zero real secrets, PII, or private keys."
            ),
            "false": (
                "Unsafe: Contains real API keys, private keys, passwords, internal IP addresses, "
                "customer PII (PESEL/SSN/email/phone), or confidential company trade secrets."
            ),
        },
    },
    "contains_censorship_triggers": {
        "type": "noul",
        "instructions": (
            "Does this prompt contain security exploit, reverse-engineering, vulnerability payload, "
            "or sensitive legal/medical terms that could trigger refusal guardrails on a commercial API?"
        ),
        "criteria": {
            "true": (
                "Yes: Mentions exploits, shellcode, malware, bypassing authentication, reverse engineering, "
                "or raw vulnerability payloads that should be decomposed into neutral algorithmic micro-tasks."
            ),
            "false": (
                "No: Uses neutral software engineering, mathematical, or algorithmic terminology."
            ),
        },
    },
    "delegation_decision": {
        "type": "choice",
        "instructions": (
            "Assuming the external API logs and trains on all inputs, how should the local Octop "
            "orchestrator handle this request?"
        ),
        "criteria": {
            "delegate_to_external_api": (
                "Safe and abstract task; approve direct delegation to the external frontier API."
            ),
            "sanitize_and_decompose_microtasks": (
                "Contains PII/identifiers or security/exploit terminology that can be masked with "
                "synthetic tokens and decomposed into neutral algorithmic micro-tasks before delegation."
            ),
            "reject_force_local_execution": (
                "Contains critical private keys, raw production credentials, destructive commands, "
                "or non-delegable core secrets; block external API and force local execution / self-correction."
            ),
        },
    },
    "retention_exposure_severity": {
        "type": "score",
        "instructions": (
            "Rate the data-leakage risk if this exact prompt is permanently retained in an external "
            "provider's training dataset:"
        ),
        "criteria": [
            "Zero risk: Purely abstract algorithm, math, or generic public code",
            "Low risk: Sanitized logic using synthetic <MOCK_...> tokens only",
            "Moderate risk: Internal network topology, business logic, or security audit context",
            "Critical risk: Unmasked API keys, cryptographic private keys, customer PII, or destructive commands",
        ],
    },
}


@dataclass
class OctopSafetyDecision:
    approved_for_external_api: bool
    decision_action: str
    decision_confidence: float
    safe_for_retention_prob: float
    censorship_trigger_prob: float
    retention_severity_score: float
    reason_code: str
    feedback_for_orchestrator: Optional[str]
    input_tokens: int
    output_tokens: int
    inference_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


def resolve_local_d1_path(model_id_or_path: str = DEFAULT_D1_REPO) -> str:
    """Resolves cached snapshot directory for `LiquidAI/d1-3B-w8a8` if available locally."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    patterns = [
        "/mnt/d/hf_cache/hub/models--LiquidAI--d1-3B-w8a8/snapshots/*",
        os.path.expanduser("~/.cache/huggingface/hub/models--LiquidAI--d1-3B-w8a8/snapshots/*"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        for m in matches:
            if os.path.isfile(os.path.join(m, "model.safetensors")):
                return m
    return model_id_or_path


class LiquidD1DecisionEngine:
    """Manages the real `LiquidAI/d1-3B-w8a8` INT8 W8A8 decision model for Octop API safety checks."""

    CRITICAL_LOCAL_ONLY_MARKERS = (
        "BEGIN RSA PRIVATE KEY",
        "BEGIN OPENSSH PRIVATE KEY",
        "BEGIN EC PRIVATE KEY",
        "rm -rf /",
        "mkfs.",
        ":(){ :|:& };:",
    )

    def __init__(
        self,
        model_id_or_path: str = DEFAULT_D1_REPO,
        device: str = "cuda:0",
        compile_model: bool = False,
    ) -> None:
        apply_torchao_d1_compat_shim()
        self.resolved_path = resolve_local_d1_path(model_id_or_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        from transformers import AutoModel

        t0 = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            self.resolved_path,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        if compile_model and self.device.type == "cuda":
            self.model.compile(mode="reduce-overhead")
        self.load_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    @property
    def language_layers(self) -> nn.ModuleList:
        """Returns the 30 hybrid (`conv` + `full_attention` + `MLP`) layers of `LiquidAI/d1-3B-w8a8`."""
        return self.model.model.language_model.layers

    def evaluate_octop_delegation(
        self,
        candidate_prompt: str | Mapping[str, Any],
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> OctopSafetyDecision:
        """Runs a single-pass System One evaluation (`output_tokens=0`) on `LiquidAI/d1-3B-w8a8`."""
        state_obj: Any
        if extra_context:
            state_obj = {
                "candidate_external_api_payload": candidate_prompt,
                "external_api_policy": {
                    "retains_prompts_permanently": True,
                    "trains_frontier_models_on_inputs": True,
                    **dict(extra_context),
                },
            }
        else:
            state_obj = candidate_prompt

        t0 = time.perf_counter()
        raw_out = self.model.system_one(state_obj, OCTOP_EXTERNAL_API_QUESTIONS)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return self._parse_d1_output(
            raw_out=raw_out,
            candidate_text=str(candidate_prompt),
            elapsed_ms=elapsed_ms,
        )

    def evaluate_octop_batch(
        self,
        candidate_prompts: Sequence[str | Mapping[str, Any]],
    ) -> List[OctopSafetyDecision]:
        """Evaluates multiple candidate subtasks packed into a single batch pass with no padding."""
        t0 = time.perf_counter()
        requests = [(p, OCTOP_EXTERNAL_API_QUESTIONS) for p in candidate_prompts]
        batch_out = self.model.system_one_batch(requests)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / max(1, len(candidate_prompts))

        return [
            self._parse_d1_output(
                raw_out=out,
                candidate_text=str(prompt),
                elapsed_ms=per_item_ms,
            )
            for prompt, out in zip(candidate_prompts, batch_out)
        ]

    def evaluate_octop_superbatch_packed(
        self,
        candidate_prompts: Sequence[str],
    ) -> Tuple[List[OctopSafetyDecision], Dict[str, float]]:
        """Packs `B` prompts x 4 Octop safety questions into ONE zero-padding Tensor Core forward pass.

        Unlike upstream `system_one_batch` (which runs multi-question states sequentially one-by-one),
        this method flattens all `(state_i, question_j)` pairs into a single packed `Tree` pass so
        `torch._int_mm` INT8 Tensor Cores receive a massive `M = sum(tokens)` matrix in one shot.
        """
        if not candidate_prompts:
            return [], {}

        t0 = time.perf_counter()
        q_names = list(OCTOP_EXTERNAL_API_QUESTIONS.keys())
        single_q_requests = [
            (prompt, {q_name: OCTOP_EXTERNAL_API_QUESTIONS[q_name]})
            for prompt in candidate_prompts
            for q_name in q_names
        ]
        packed_results = self.model.system_one_batch(single_q_requests)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        num_q = len(q_names)
        decisions: List[OctopSafetyDecision] = []
        total_input_tokens = 0

        for i, prompt in enumerate(candidate_prompts):
            merged_answers: Dict[str, Any] = {}
            prompt_tokens = 0
            for j, q_name in enumerate(q_names):
                res_ij = packed_results[i * num_q + j]
                merged_answers[q_name] = res_ij["answers"][q_name]
                prompt_tokens += int(res_ij["usage"]["input_tokens"])
            total_input_tokens += prompt_tokens
            merged_raw = {
                "answers": merged_answers,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
            }
            decisions.append(
                self._parse_d1_output(
                    raw_out=merged_raw,
                    candidate_text=str(prompt),
                    elapsed_ms=elapsed_ms / len(candidate_prompts),
                )
            )

        # 5.23 GOPS per token across 30 hybrid INT8 W8A8 layers + lm_head
        total_int8_ops = total_input_tokens * 5.23e9
        int8_tops = (total_int8_ops / (elapsed_ms / 1000.0)) / 1e12
        tokens_per_sec = total_input_tokens / (elapsed_ms / 1000.0)
        decisions_per_sec = (len(candidate_prompts) * num_q) / (elapsed_ms / 1000.0)
        prompts_per_sec = len(candidate_prompts) / (elapsed_ms / 1000.0)

        telemetry = {
            "batch_prompts": float(len(candidate_prompts)),
            "total_questions_evaluated": float(len(candidate_prompts) * num_q),
            "total_packed_tokens": float(total_input_tokens),
            "wall_time_ms": round(elapsed_ms, 2),
            "throughput_tokens_per_sec": round(tokens_per_sec, 1),
            "throughput_prompts_per_sec": round(prompts_per_sec, 2),
            "throughput_decisions_per_sec": round(decisions_per_sec, 2),
            "sustained_int8_tensor_core_tops": round(int8_tops, 2),
        }
        return decisions, telemetry

    def _parse_d1_output(
        self,
        raw_out: Dict[str, Any],
        candidate_text: str,
        elapsed_ms: float,
    ) -> OctopSafetyDecision:
        answers = raw_out.get("answers", {})
        usage = raw_out.get("usage", {})

        safe_prob = float(answers.get("safe_for_external_retention", {}).get("noul", 0.5))
        censor_prob = float(answers.get("contains_censorship_triggers", {}).get("noul", 0.0))
        choice_info = answers.get("delegation_decision", {})
        action = str(choice_info.get("choice", "sanitize_and_decompose_microtasks"))
        confidence = float(choice_info.get("confidence", 0.5))
        severity = float(answers.get("retention_exposure_severity", {}).get("score", 1.5))

        # Check critical non-delegable markers (cryptographic private keys / destructive shell)
        has_critical_marker = any(m in candidate_text for m in self.CRITICAL_LOCAL_ONLY_MARKERS)
        if has_critical_marker:
            action = "reject_force_local_execution"
            safe_prob = min(safe_prob, 0.01)
            severity = max(severity, 2.95)

        approved = (
            action == "delegate_to_external_api"
            and safe_prob >= 0.60
            and not has_critical_marker
        )

        if approved:
            reason_code = "OCTOP_SAFE_FOR_EXTERNAL_RETENTION"
            feedback = None
        elif action == "reject_force_local_execution" or has_critical_marker:
            if has_critical_marker:
                reason_code = "CRITICAL_PRIVATE_KEY_OR_DESTRUCTIVE_PAYLOAD"
            else:
                reason_code = "UNSAFE_FOR_EXTERNAL_API_RETENTION_AND_TRAINING"
            action = "reject_force_local_execution"
            feedback = (
                f"Odrzucono delegację do zewnętrznego API (zakładana retencja i trening na danych): "
                f"wykryto {reason_code} (P(safe)={safe_prob:.3f}, severity={severity:.2f}/3.00). "
                f"Usuń klucze/PII lub wygeneruj czysto algorytmiczną abstrakcję zadania."
            )
        else:
            reason_code = (
                "CENSORSHIP_TRIGGER_REQUIRES_DECOMPOSITION"
                if censor_prob >= 0.50
                else "SENSITIVE_CONTEXT_REQUIRES_SANITIZATION"
            )
            action = "sanitize_and_decompose_microtasks"
            feedback = (
                f"Zadanie wymaga sanityzacji PII/sekretów (<MOCK_...>) oraz dekompozycji na neutralne "
                f"mikro-zadania przed wysyłką do zewnętrznego API ({reason_code}, P(safe)={safe_prob:.3f})."
            )

        return OctopSafetyDecision(
            approved_for_external_api=approved,
            decision_action=action,
            decision_confidence=round(confidence, 4),
            safe_for_retention_prob=round(safe_prob, 4),
            censorship_trigger_prob=round(censor_prob, 4),
            retention_severity_score=round(severity, 4),
            reason_code=reason_code,
            feedback_for_orchestrator=feedback,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            inference_ms=round(elapsed_ms, 2),
            raw_answers=answers,
        )
