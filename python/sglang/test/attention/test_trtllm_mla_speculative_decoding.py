import unittest

import torch
import numpy as np
from types import SimpleNamespace

from sglang.srt.utils import is_flashinfer_available
from sglang.srt.layers.dp_attention import get_attention_tp_size as _get_attn_tp_size
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

# Re-use the compare util from the existing TRTLLM MLA tests
from sglang.test.attention.test_trtllm_mla_backend import compare_outputs

# Backends under test
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAMultiStepDraftBackend,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "device": "cuda",
    "dtype": torch.bfloat16,
    "kv_cache_dtype": torch.bfloat16,
    "context_len": 2048,
    "num_attention_heads": 128,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "num_kv_heads": 1,
    "layer_id": 0,
    "page_size": 32,
    "max_bs": 16,
    "seed_cache": 42,
    "seed_qkv": 123,
}

TOPK = 1
SPEC_STEPS = 2
TOLERANCE = 1e-2

# -----------------------------------------------------------------------------
# Helper mocks
# -----------------------------------------------------------------------------
class MockModelRunner:
    """Minimal fake ModelRunner with server_args for speculative backends."""

    def __init__(self, cfg):
        # Basic attrs
        self.device = cfg["device"]
        self.dtype = cfg["dtype"]
        self.kv_cache_dtype = cfg["kv_cache_dtype"]
        self.page_size = cfg["page_size"]

        # server_args stub – only fields accessed by multi-step backends
        self.server_args = SimpleNamespace(
            page_size=cfg["page_size"],
            speculative_eagle_topk=TOPK,
            speculative_num_steps=SPEC_STEPS,
        )

        # Model config stub
        scaling = 1.0 / ((cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]) ** 0.5)
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": cfg["context_len"],
                "attention_arch": 3,  # AttentionArch.MLA enum value not needed
                "num_attention_heads": cfg["num_attention_heads"],
                "kv_lora_rank": cfg["kv_lora_rank"],
                "qk_nope_head_dim": cfg["qk_nope_head_dim"],
                "qk_rope_head_dim": cfg["qk_rope_head_dim"],
                "v_head_dim": cfg["v_head_dim"],
                "scaling": scaling,
                "get_num_kv_heads": staticmethod(lambda _: cfg["num_kv_heads"]),
            },
        )

        # Req-to-token pool
        max_bs = cfg["max_bs"]
        max_ctx = cfg["context_len"]
        req_to_token = torch.arange(
            max_bs * max_ctx, dtype=torch.int32, device=self.device
        ).reshape(max_bs, max_ctx)
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_bs,
                "req_to_token": req_to_token,
            },
        )

        # KV cache pool (MLA)
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_bs * max_ctx,
            page_size=cfg["page_size"],
            dtype=self.kv_cache_dtype,
            kv_lora_rank=cfg["kv_lora_rank"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )

        # For API compatibility
        self.tp_size = 1
        self.tp_rank = 0
        self.dp_size = 1
        self.use_mla_backend = True

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def populate_kv_cache(batch_size, seq_lens, model_runners, layer, cfg):
    """Populate identical MLA KV cache for provided model runners."""
    torch.manual_seed(cfg["seed_cache"])
    for mr in model_runners:
        torch.manual_seed(cfg["seed_cache"])
        for i in range(batch_size):
            seq_len = int(seq_lens[i].item())
            for token_idx in range(seq_len - 1):
                cache_k_nope = torch.randn(
                    (1, cfg["qk_nope_head_dim"]), dtype=cfg["dtype"], device=cfg["device"]
                )
                cache_k_rope = torch.randn(
                    (1, cfg["qk_rope_head_dim"]), dtype=cfg["dtype"], device=cfg["device"]
                )
                cache_loc = mr.req_to_token_pool.req_to_token[i, token_idx]
                mr.token_to_kv_pool.set_mla_kv_buffer(
                    layer,
                    cache_loc.unsqueeze(0),
                    cache_k_nope.squeeze(0),
                    cache_k_rope.squeeze(0),
                )


def create_qkv_tensors(batch_size, cfg):
    head_dim = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
    q = torch.randn(
        (batch_size, cfg["num_attention_heads"], head_dim),
        dtype=cfg["dtype"],
        device=cfg["device"],
    )
    k = torch.randn(
        (batch_size, cfg["num_kv_heads"], head_dim),
        dtype=cfg["dtype"],
        device=cfg["device"],
    )
    v = torch.randn(
        (batch_size, cfg["num_kv_heads"], cfg["v_head_dim"]),
        dtype=cfg["dtype"],
        device=cfg["device"],
    )
    return q, k, v


def create_forward_batch(batch_size, seq_lens, model_runner, positions, spec_info, cfg):
    fb = ForwardBatch(
        batch_size=batch_size,
        input_ids=torch.randint(0, 100, (batch_size, 1), device=cfg["device"]),
        out_cache_loc=torch.arange(batch_size, device=cfg["device"]),
        seq_lens_sum=int(seq_lens.sum().item()),
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.arange(batch_size, device=cfg["device"]),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        positions=positions,
        attn_backend=None,
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        spec_info=spec_info,
    )
    fb.req_to_token_pool = model_runner.req_to_token_pool
    fb.token_to_kv_pool = model_runner.token_to_kv_pool
    return fb

# -----------------------------------------------------------------------------
# Test Case
# -----------------------------------------------------------------------------

@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer required",
)
class TestTRTLLMMLASpeculativeDecoding(unittest.TestCase):
    def test_spec_decode_output_match(self):
        """Compare TRTLLM vs FlashInfer multi-step draft backends outputs."""

        cfg = DEFAULT_CONFIG.copy()
        batch_size = 2
        max_seq_len = 32

        # Create model runners
        mr_trt = MockModelRunner(cfg)
        mr_ref = MockModelRunner(cfg)

        # Instantiate multi-step draft backends
        trt_multi = TRTLLMMLAMultiStepDraftBackend(mr_trt, TOPK, SPEC_STEPS)
        ref_multi = FlashInferMLAMultiStepDraftBackend(mr_ref, TOPK, SPEC_STEPS)

        # Create RadixAttention layer (shared dims)
        layer = RadixAttention(
            num_heads=cfg["num_attention_heads"],
            head_dim=cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"],
            scaling=1.0 / ((cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]) ** 0.5),
            num_kv_heads=cfg["num_kv_heads"],
            layer_id=cfg["layer_id"],
            v_head_dim=cfg["v_head_dim"],
            prefix="attn_mqa",
        )

        # Sequence lengths (varied)
        torch.manual_seed(cfg["seed_cache"])
        seq_lens = torch.randint(8, max_seq_len, (batch_size,), device=cfg["device"])
        seq_lens[0] = max_seq_len

        # Populate KV cache identically
        populate_kv_cache(batch_size, seq_lens, [mr_trt, mr_ref], layer, cfg)

        # Positions tensor (dummy)
        positions = torch.zeros(int(seq_lens.sum().item()), dtype=torch.int64, device=cfg["device"])

        # Spec info placeholder
        spec_info_trt = EagleDraftInput()
        spec_info_ref = EagleDraftInput()

        # Build forward batches
        fb_trt = create_forward_batch(batch_size, seq_lens.clone(), mr_trt, positions, spec_info_trt, cfg)
        fb_ref = create_forward_batch(batch_size, seq_lens.clone(), mr_ref, positions, spec_info_ref, cfg)

        # Initialize metadata
        trt_multi.init_forward_metadata(fb_trt)
        ref_multi.init_forward_metadata(fb_ref)

        # Use first speculative step backend for comparison
        backend_trt_step0 = trt_multi.attn_backends[0]
        backend_ref_step0 = ref_multi.attn_backends[0]

        # Create Q, K, V
        torch.manual_seed(cfg["seed_qkv"])
        q, k, v = create_qkv_tensors(batch_size, cfg)

        # Run decode
        out_trt = backend_trt_step0.forward_decode(q, k, v, layer, fb_trt)
        out_ref = backend_ref_step0.forward_decode(q.clone(), k.clone(), v.clone(), layer, fb_ref)

        # Compare
        self.assertTrue(
            compare_outputs(out_trt, out_ref, tolerance=TOLERANCE),
            "Speculative decode outputs differ beyond tolerance",
        )


if __name__ == "__main__":
    unittest.main() 