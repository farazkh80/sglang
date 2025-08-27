#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server \
  --model nvidia/DeepSeek-R1-0528-FP4 \
  --trust-remote-code --quantization modelopt_fp4 \
  --attention-backend trtllm_mla --kv-cache-dtype fp8_e4m3 \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 40000
