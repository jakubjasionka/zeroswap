#!/usr/bin/env bash
# Launch vLLM with ZeroSwap Worker Extension + ZeroSwap OpenAI Interceptor Control Plane
set -euo pipefail

PRIMARY_MODEL="${PRIMARY_MODEL:-/mnt/d/hf_cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/e8f3c7c4de75ad84fe1bcef95d38eca76214480b}"
SECONDARY_MODEL="${SECONDARY_MODEL:-lfm-1b}"
VLLM_PORT="${VLLM_PORT:-8000}"
ZEROSWAP_PORT="${ZEROSWAP_PORT:-8080}"
GPU_INDEX="${GPU_INDEX:-0}"

echo "================================================================"
echo " ZeroSwap + vLLM Launcher"
echo " Primary Orchestrator : $PRIMARY_MODEL"
echo " Secondary Guardrail  : $SECONDARY_MODEL"
echo " vLLM Engine Port     : $VLLM_PORT"
echo " ZeroSwap Proxy Port  : $ZEROSWAP_PORT"
echo "================================================================"

# Step 1: Run Zero-Config Auto-Tuning Profiler
python3 -m zeroswap.tune \
  --primary "$PRIMARY_MODEL" \
  --secondary "$SECONDARY_MODEL" \
  --gpu "$GPU_INDEX" \
  --output plan.json

# Step 2: Start ZeroSwap OpenAI Control-Plane Interceptor
echo "Starting ZeroSwap OpenAI-compatible Control Plane on :$ZEROSWAP_PORT ..."
exec python3 -m zeroswap.vllm_integration.openai_server \
  --host 0.0.0.0 \
  --port "$ZEROSWAP_PORT" \
  --upstream-vllm "http://localhost:$VLLM_PORT" \
  --primary "$PRIMARY_MODEL" \
  --secondary "$SECONDARY_MODEL" \
  --gpu "$GPU_INDEX"
