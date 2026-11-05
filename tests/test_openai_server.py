"""Unit & integration tests for the ZeroSwap OpenAI-compatible vLLM Control-Plane Server."""

from __future__ import annotations

import asyncio
import httpx

from zeroswap.vllm_integration.openai_server import create_zeroswap_app


def test_openai_server_endpoints() -> None:
    app = create_zeroswap_app(primary_model="qwen-32b", secondary_model="lfm-1b", gpu_index=0)

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Health check
            r_health = await client.get("/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "ok"

            # 2. Models endpoint
            r_models = await client.get("/v1/models")
            assert r_models.status_code == 200
            models_data = r_models.json()["data"]
            assert len(models_data) == 2

            # 3. ZeroSwap status
            r_status = await client.get("/v1/zeroswap/status")
            assert r_status.status_code == 200
            assert r_status.json()["plan"]["kv_cache_max_tokens"] > 0

            # 4. Standalone Guardrail Verify endpoint
            r_verify = await client.post(
                "/v1/zeroswap/verify",
                json={
                    "prompt": "Connect to 10.10.1.5 with token sk-abcdefgh1234567890",
                    "total_tokens": 4096,
                    "cached_tokens": 1000,
                },
            )
            assert r_verify.status_code == 200
            v_data = r_verify.json()
            assert v_data["approved"] is True
            assert v_data["redacted_entities"] == 2
            assert "<MOCK_INTERNAL_IP_1>" in v_data["sanitized_prompt"]

            # 5. OpenAI Chat Completions with automatic PII masking, unmasking, and telemetry headers
            r_chat = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen-32b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Check server 192.168.55.10 using key sk-11223344556677889900",
                        }
                    ],
                    "cached_tokens": 500,
                },
            )
            assert r_chat.status_code == 200
            assert r_chat.headers["X-ZeroSwap-KV-Invariant"] == "preserved"
            assert int(r_chat.headers["X-ZeroSwap-Redacted-Entities"]) == 2
            chat_data = r_chat.json()
            content = chat_data["choices"][0]["message"]["content"]
            # Real IP and key are unmasked in the final local response
            assert "192.168.55.10" in content
            assert "sk-11223344556677889900" in content

    asyncio.run(_run())
