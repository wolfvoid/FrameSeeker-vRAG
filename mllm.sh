#!/bin/bash
export CUDA_VISIBLE_DEVICES=8,9
export VLLM_USE_V1=0
# 启动 Server
# --tensor-parallel-size 2: 使用两张卡并行
# --host 0.0.0.0: 允许外部访问
python -m vllm.entrypoints.openai.api_server \
    --model /data3/meibingyin/models/Qwen/Qwen3-VL-8B-Instruct \
    --served-model-name qwen-vl \
    --trust-remote-code \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --limit-mm-per-prompt '{"image": 5}'
