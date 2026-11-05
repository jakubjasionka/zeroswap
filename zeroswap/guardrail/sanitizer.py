"""DataSanitizationGate: Local PII/Secret Masking, Unmasking & Anti-Censorship Micro-Task Splitter.

Implements the Hybrid Control Plane privacy architecture:
1. Detects API keys, tokens, internal IPv4 addresses, emails, phone numbers, and national IDs (PESEL/SSN).
2. Replaces sensitive values with deterministic synthetic mocks (`<MOCK_API_KEY_1>`, `<MOCK_IP_1>`, etc.)
   before any external frontier API call.
3. Restores (`unmask`) real data locally when the frontier response returns.
4. Decomposes sensitive domain queries (security audit, reverse engineering, legal/medical analysis)
   into neutral algorithmic micro-tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


@dataclass
class SanitizationResult:
    sanitized_text: str
    mock_to_real: Dict[str, str] = field(default_factory=dict)
    detected_categories: List[str] = field(default_factory=list)
    redacted_count: int = 0


class DataSanitizationGate:
    """Reversible local PII & credential sanitizer and micro-task decomposer."""

    PATTERNS: List[Tuple[str, Pattern[str]]] = [
        (
            "API_KEY",
            re.compile(
                r"\b(?:sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9\-._~+/]+=*)\b"
            ),
        ),
        (
            "INTERNAL_IP",
            re.compile(
                r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b"
            ),
        ),
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
        (
            "PESEL_OR_SSN",
            re.compile(r"\b(?:\d{11}|\d{3}-\d{2}-\d{4})\b"),
        ),
        (
            "PHONE",
            re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,3}\)[\s-]?)?\b\d{3}[\s-]?\d{3}[\s-]?\d{3}\b"),
        ),
    ]

    DOMAIN_TRIGGER_MAP: Dict[str, str] = {
        "exploit": "boundary_condition_input",
        "vulnerability": "state_transition_anomaly",
        "reverse engineer": "reconstruct_state_machine",
        "payload": "serialized_data_frame",
        "bypass": "alternative_control_flow",
        "shellcode": "byte_sequence_buffer",
        "malware": "untrusted_binary_artifact",
    }

    def sanitize(self, text: str) -> SanitizationResult:
        """Masks all PII, secrets, and internal IPs with reversible synthetic mock tokens."""
        mock_to_real: Dict[str, str] = {}
        real_to_mock: Dict[str, str] = {}
        categories: List[str] = []
        counters: Dict[str, int] = {}

        sanitized = text
        for category, pattern in self.PATTERNS:
            def _replacer(match: re.Match[str]) -> str:
                raw_val = match.group(0)
                if raw_val in real_to_mock:
                    return real_to_mock[raw_val]
                idx = counters.get(category, 0) + 1
                counters[category] = idx
                mock_token = f"<MOCK_{category}_{idx}>"
                real_to_mock[raw_val] = mock_token
                mock_to_real[mock_token] = raw_val
                if category not in categories:
                    categories.append(category)
                return mock_token

            sanitized = pattern.sub(_replacer, sanitized)

        return SanitizationResult(
            sanitized_text=sanitized,
            mock_to_real=mock_to_real,
            detected_categories=categories,
            redacted_count=len(mock_to_real),
        )

    def unmask(self, text: str, mock_to_real: Dict[str, str]) -> str:
        """Restores synthetic mocks back to real business/local values after frontier execution."""
        restored = text
        for mock_token, real_value in mock_to_real.items():
            restored = restored.replace(mock_token, real_value)
        return restored

    def decompose_into_abstract_microtasks(self, prompt: str, max_tasks: int = 4) -> List[str]:
        """Neutralizes domain-sensitive terminology and decomposes a query into abstract micro-tasks."""
        abstracted = prompt
        for trigger, neutral in self.DOMAIN_TRIGGER_MAP.items():
            abstracted = re.sub(
                re.escape(trigger), neutral, abstracted, flags=re.IGNORECASE
            )

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", abstracted)
            if s.strip()
        ]
        if not sentences:
            return [abstracted]

        chunk_size = max(1, (len(sentences) + max_tasks - 1) // max_tasks)
        microtasks: List[str] = []
        for i in range(0, len(sentences), chunk_size):
            segment = " ".join(sentences[i : i + chunk_size])
            task_idx = len(microtasks) + 1
            microtasks.append(
                f"[MicroTask #{task_idx} - Pure Algorithmic Specification]: {segment}"
            )
        return microtasks[:max_tasks]
