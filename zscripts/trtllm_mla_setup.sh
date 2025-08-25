#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server \
  --model nvidia/DeepSeek-R1-0528-FP4 \
  --trust-remote-code --quantization modelopt_fp4 --enable-flashinfer-cutlass-moe --enable-ep-moe \
  --attention-backend trtllm_mla --kv-cache-dtype fp8_e4m3 \
  --tp-size 4 --dp-size 4 --enable-dp-attention --mem-fraction-static 0.83 --chunked-prefill-size 8192 \
  --max-running-requests 1024 --cuda-graph-max-bs 256 \
  --enable-ep-moe \
  --host 0.0.0.0 \
  --port 40000 \
