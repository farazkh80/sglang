CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 -m sglang.launch_server \
  --model nvidia/DeepSeek-R1-0528-FP4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --enable-flashinfer-cutlass-moe \
  --enable-ep-moe \
  --attention-backend flashinfer \
  --tp-size 4 \
  --dp-size 4 \
  --enable-dp-attention \
  --mem-fraction-static 0.83 \
  --chunked-prefill-size 8192 \
  --max-running-requests 1024 \
  --cuda-graph-max-bs 256 \
  --host 0.0.0.0 \
  --port 30001 