# python -m venv nv_eval
source nv_eval/bin/activate

# pip install git+https://github.com/NVIDIA/NeMo-Skills.git


# # prepare the GPQA dataset (this is what's missing!)
# python -m nemo_skills.dataset.prepare gpqa

PORT=30001
Backend=trtllm_mla_full_cuda_graph_off_toms_server

ns eval \
  --server_type=openai \
  --model=nvidia/DeepSeek-R1-0528-FP4 \
  --server_address=http://localhost:${PORT}/v1 \
  --benchmarks=gpqa:0 \
  --output_dir=./nemo_skills_output_${Backend}_$(date +%Y%m%d_%H%M%S) \
  ++max_concurrent_requests=10000 \
  ++server.api_key=dummy \
  ++inference.tokens_to_generate=16384