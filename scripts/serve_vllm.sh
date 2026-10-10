#!/bin/bash
# The scheduler / verifier VLM: Qwen3-VL-8B-Instruct-FP8 on vLLM, one request at a time (as every experiment ran).
#   scripts/serve_vllm.sh [GPU] [PORT]            e.g. scripts/serve_vllm.sh 0 8010
# MODEL: a local path or HF id (default Qwen/Qwen3-VL-8B-Instruct-FP8). UTIL: GPU memory share (default 0.715 leaves
# ~4 GB of a 16 GB card for the simulator, detector and SAM). LORA=1: serve evolved verifiers beside the base, loaded
# at runtime (tools/verifier_evolve.py load); the LoRA buffers need a little more memory (UTIL=0.76 on a 16 GB card
# with the simulator on another GPU).
GPU=${1:-0}; PORT=${2:-8010}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}
X=""
if [ -n "$LORA" ]; then X="--enable-lora --max-lora-rank 16 --max-loras 1"; export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True; fi
CUDA_VISIBLE_DEVICES=$GPU exec vllm serve "$MODEL" --served-model-name qwen3vl \
  --gpu-memory-utilization ${UTIL:-0.715} --max-model-len 2560 --max-num-seqs 1 --max-num-batched-tokens 2560 \
  --enforce-eager --limit-mm-per-prompt '{"image":2}' --mm-processor-kwargs '{"max_pixels": 602112}' --port $PORT $X
