#!/usr/bin/env python3
"""
Compare MLA attention debug dumps between flashinfer and TRTLLM backends.

Inputs expected from backends (per step):

- flashinfer (from FlashInferMLAAttnBackend.forward_decode with SGLANG_MLA_DEBUG_FLASHINFER=1):
  - q_nope.pt        [bs, num_heads, v_head_dim] or [seq_len, heads, dim] flattened via view
  - q_rope.pt        [bs, num_heads, qk_rope_head_dim]
  - k_buf.pt         [total_pages, page_size, kv_dim] (paged KV buffer view)
  - attn_out.pt      [bs, num_heads*v_head_dim]

- trtllm (from TRTLLMMLABackend.forward_decode with SGLANG_MLA_DEBUG_TRTLLM=1):
  - query.pt         Merged query fed to TRT kernel. If FP16 path, also q_nope.pt and q_rope.pt exist
  - q_nope.pt        Optional, only on FP16 path
  - q_rope.pt        Optional, only on FP16 path
  - kv_cache.pt      [bs, 1, page_size, kv_dim] paged view used by TRT kernel
  - attn_out.pt      [bs, num_heads*v_head_dim]

We align tensor names and compute detailed diff stats.
"""

import argparse
import os
from pathlib import Path
import json
import torch


def load_tensor(path: Path):
    try:
        if not path.exists():
            return None
        return torch.load(str(path), map_location="cpu")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.reshape(-1).float()
    b32 = b.reshape(-1).float()
    if a32.numel() == 0 or b32.numel() == 0:
        return float("nan")
    return torch.nn.functional.cosine_similarity(a32, b32, dim=0).item()


def _safe_percentiles(flat: torch.Tensor, percentiles=(0.50, 0.90, 0.95, 0.99),
                      max_elements: int = 5_000_000, sample_size: int = 1_000_000):
    """Compute percentiles safely.

    If the tensor is huge, compute on a strided sample to avoid quantile kernel limits.
    """
    if flat.numel() == 0:
        return {p: 0.0 for p in percentiles}
    try:
        if flat.numel() <= max_elements:
            q = torch.quantile(flat, torch.tensor(percentiles, dtype=torch.float32))
            return {float(p): q[i].item() for i, p in enumerate(percentiles)}
        # Downsample by stride or cap at sample_size
        step = max(1, flat.numel() // sample_size)
        sampled = flat[::step].contiguous()
        q = torch.quantile(sampled, torch.tensor(percentiles, dtype=torch.float32))
        return {float(p): q[i].item() for i, p in enumerate(percentiles)}
    except Exception:
        # Fallback: return zeros if quantile still fails
        return {p: 0.0 for p in percentiles}


def diff_stats(a: torch.Tensor, b: torch.Tensor):
    a32 = a.float()
    b32 = b.float()
    d = (a32 - b32).abs()
    flat = d.reshape(-1)
    percs = _safe_percentiles(flat)
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs_diff": d.max().item() if d.numel() > 0 else 0.0,
        "mean_abs_diff": d.mean().item() if d.numel() > 0 else 0.0,
        "cosine_similarity": cosine_similarity(a, b),
        "p50": percs.get(0.50, 0.0),
        "p90": percs.get(0.90, 0.0),
        "p95": percs.get(0.95, 0.0),
        "p99": percs.get(0.99, 0.0),
    }


def print_head(t: torch.Tensor, name: str, n: int = 10):
    vals = t.reshape(-1)[:n].tolist()
    print(f"    {name}[:{n}]: {vals}")


def align_q_parts(
    fi_q_nope: torch.Tensor,
    fi_q_rope: torch.Tensor,
    trt_query: torch.Tensor,
    trt_q_nope: torch.Tensor | None,
    trt_q_rope: torch.Tensor | None,
):
    """
    Return aligned tuples: (q_nope_fi, q_nope_trt, q_rope_fi, q_rope_trt)
    Handle cases where trt dumps only merged 'query'.
    """
    q_nope_fi = fi_q_nope
    q_rope_fi = fi_q_rope

    if trt_q_nope is not None and trt_q_rope is not None:
        q_nope_trt = trt_q_nope
        q_rope_trt = trt_q_rope
    else:
        # Split merged query into nope/rope assuming last dim = v_head_dim + qk_rope_head_dim
        assert trt_query is not None, "TRT dump missing 'query.pt'"
        last_dim = trt_query.shape[-1]
        v_dim = fi_q_nope.shape[-1]
        q_nope_trt = trt_query[..., :v_dim]
        q_rope_trt = trt_query[..., v_dim:]

    return q_nope_fi, q_nope_trt.contiguous(), q_rope_fi, q_rope_trt.contiguous()


def compare_step(step_dir_fi: Path, step_dir_trt: Path, threshold: float, kv_max_elements: int, kv_sample_pages: int, do_kv_compare: bool):
    results = {"tensors": {}, "summary": {}}

    # Load flashinfer tensors
    fi_q_nope = load_tensor(step_dir_fi / "q_nope.pt")
    fi_q_rope = load_tensor(step_dir_fi / "q_rope.pt")
    fi_k_buf = load_tensor(step_dir_fi / "k_buf.pt")
    fi_out = load_tensor(step_dir_fi / "attn_out.pt")

    # Load trtllm tensors
    trt_query = load_tensor(step_dir_trt / "query.pt")
    trt_q_nope = load_tensor(step_dir_trt / "q_nope.pt")
    trt_q_rope = load_tensor(step_dir_trt / "q_rope.pt")
    trt_k_cache = load_tensor(step_dir_trt / "kv_cache.pt")

    # Normalize TRT shapes: squeeze optional singleton dims introduced by backend
    # query: [bs, 1, heads, dim] -> [bs, heads, dim]
    if trt_query is not None and trt_query.ndim >= 3 and trt_query.shape[1] == 1:
        trt_query = trt_query.squeeze(1).contiguous()
    # q_nope/q_rope: sometimes [bs, 1, heads, dim] -> [bs, heads, dim]
    if trt_q_nope is not None and trt_q_nope.ndim >= 3 and trt_q_nope.shape[1] == 1:
        trt_q_nope = trt_q_nope.squeeze(1).contiguous()
    if trt_q_rope is not None and trt_q_rope.ndim >= 3 and trt_q_rope.shape[1] == 1:
        trt_q_rope = trt_q_rope.squeeze(1).contiguous()
    # kv_cache: [pages, 1, page_size, kv_dim] -> [pages, page_size, kv_dim]
    if trt_k_cache is not None and trt_k_cache.ndim >= 3 and trt_k_cache.shape[1] == 1:
        trt_k_cache = trt_k_cache.squeeze(1).contiguous()
    trt_out = load_tensor(step_dir_trt / "attn_out.pt")

    print("  Inputs present:")
    for name, t in [
        ("FI q_nope", fi_q_nope),
        ("FI q_rope", fi_q_rope),
        ("FI k_buf", fi_k_buf),
        ("TRT query", trt_query),
        ("TRT q_nope", trt_q_nope),
        ("TRT q_rope", trt_q_rope),
        ("TRT kv_cache", trt_k_cache),
    ]:
        print(f"    {name}: {None if t is None else list(t.shape)}")

    # Compare q parts
    if fi_q_nope is not None and fi_q_rope is not None and (
        trt_query is not None or (trt_q_nope is not None and trt_q_rope is not None)
    ):
        qn_fi, qn_trt, qr_fi, qr_trt = align_q_parts(
            fi_q_nope, fi_q_rope, trt_query, trt_q_nope, trt_q_rope
        )
        # Ensure same shape
        if qn_fi.shape == qn_trt.shape:
            st = diff_stats(qn_fi, qn_trt)
            results["tensors"]["q_nope"] = st
            print(
                f"  Q NoPE: max={st['max_abs_diff']:.2e} mean={st['mean_abs_diff']:.2e} cos={st['cosine_similarity']:.6f}"
            )
            if st["max_abs_diff"] > threshold:
                print("    ⚠️ exceeds threshold")
        else:
            results["tensors"]["q_nope"] = {"error": f"shape mismatch {qn_fi.shape} vs {qn_trt.shape}"}
            print("  Q NoPE: shape mismatch")

        if qr_fi.shape == qr_trt.shape:
            st = diff_stats(qr_fi, qr_trt)
            results["tensors"]["q_rope"] = st
            print(
                f"  Q RoPE: max={st['max_abs_diff']:.2e} mean={st['mean_abs_diff']:.2e} cos={st['cosine_similarity']:.6f}"
            )
            if st["max_abs_diff"] > threshold:
                print("    ⚠️ exceeds threshold")
        else:
            results["tensors"]["q_rope"] = {"error": f"shape mismatch {qr_fi.shape} vs {qr_trt.shape}"}
            print("  Q RoPE: shape mismatch")

    # Compare outputs
    if fi_out is not None and trt_out is not None and fi_out.shape == trt_out.shape:
        st = diff_stats(fi_out, trt_out)
        results["tensors"]["attn_out"] = st
        print(
            f"  Attn Out: max={st['max_abs_diff']:.2e} mean={st['mean_abs_diff']:.2e} cos={st['cosine_similarity']:.6f}"
        )
        if st["max_abs_diff"] > threshold:
            print("    ⚠️ exceeds threshold")
    else:
        results["tensors"]["attn_out"] = {"error": "missing or shape mismatch"}
        print("  Attn Out: missing or shape mismatch")

    # ------------------------------------------------------------------
    # KV cache comparison
    # We ALWAYS flatten pages -> rows to make differences in page sizes
    # irrelevant and give a head-to-head per-token view.
    # ------------------------------------------------------------------
    if fi_k_buf is not None and trt_k_cache is not None:
        fi_shape = list(fi_k_buf.shape)
        trt_shape = list(trt_k_cache.shape)
        results["tensors"]["kv_shapes"] = {"fi": fi_shape, "trt": trt_shape}
        print(f"  KV shapes: fi {fi_shape} vs trt {trt_shape}")

        if do_kv_compare:
            # Flatten both regardless of original shape
            assert fi_k_buf.ndim == 3 and trt_k_cache.ndim == 3, "Unexpected KV tensor rank"
            fi_flat = fi_k_buf.reshape(-1, fi_k_buf.shape[-1])
            trt_flat = trt_k_cache.reshape(-1, trt_k_cache.shape[-1])

            # Align lengths
            min_len = min(fi_flat.shape[0], trt_flat.shape[0])
            fi_flat = fi_flat[:min_len]
            trt_flat = trt_flat[:min_len]

            total_elems = fi_flat.numel()
            if total_elems <= kv_max_elements:
                st = diff_stats(fi_flat, trt_flat)
                results["tensors"]["kv_cache_flat"] = st
                print(
                    f"  KV Cache (flatten): rows={min_len} max={st['max_abs_diff']:.2e} mean={st['mean_abs_diff']:.2e} cos={st['cosine_similarity']:.6f}"
                )
                if st["max_abs_diff"] > threshold:
                    print("    ⚠️ exceeds threshold")
            else:
                # sample every kth row
                stride = total_elems // kv_max_elements + 1
                fi_s = fi_flat[::stride]
                trt_s = trt_flat[::stride]
                st = diff_stats(fi_s, trt_s)
                results["tensors"]["kv_cache_flat_sample"] = {
                    **st,
                    "rows_sampled": int(fi_s.shape[0]),
                    "stride": int(stride),
                }
                print(
                    f"  KV Cache (flatten sample rows={fi_s.shape[0]} stride={stride}): max={st['max_abs_diff']:.2e} mean={st['mean_abs_diff']:.2e} cos={st['cosine_similarity']:.6f}"
                )
                if st["max_abs_diff"] > threshold:
                    print("    ⚠️ exceeds threshold")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare debug dumps between flashinfer and trtllm_mla"
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="divergence_debug",
        help="Base directory containing debug dumps",
    )
    parser.add_argument(
        "--fi-subdir",
        type=str,
        default="flashinfer/flashinfer_mla_decode",
        help="Subdir for flashinfer dumps relative to dump-dir",
    )
    parser.add_argument(
        "--trt-subdir",
        type=str,
        default="trtllm/trtllm_mla",
        help="Subdir for trtllm dumps relative to dump-dir",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1-10",
        help="Steps to compare (e.g. '1-10' or '1,3,5')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Threshold for flagging differences",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSON output file for results",
    )
    parser.add_argument(
        "--kv-compare",
        action="store_true",
        help="Enable KV cache numerical comparison (may be heavy)",
    )
    parser.add_argument(
        "--kv-max-elements",
        type=int,
        default=20_000_000,
        help="Max number of KV elements for full diff; otherwise sample",
    )
    parser.add_argument(
        "--kv-sample-pages",
        type=int,
        default=1024,
        help="Number of pages to sample from start when KV is huge",
    )

    args = parser.parse_args()
    base = Path(args.dump_dir)
    fi_base = base / args.fi_subdir
    trt_base = base / args.trt_subdir

    if not fi_base.exists():
        print(f"Error: flashinfer dir not found: {fi_base}")
        return 1
    if not trt_base.exists():
        print(f"Error: trtllm dir not found: {trt_base}")
        return 1

    if "-" in args.steps:
        a, b = args.steps.split("-")
        steps = list(range(int(a), int(b) + 1))
    else:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    all_results = []
    print(f"Comparing flashinfer vs trtllm_mla")
    print(f"Dump dir: {base}")
    print(f"Steps: {steps}")

    for step in steps:
        step_dir_fi = fi_base / f"step_{step}"
        step_dir_trt = trt_base / f"step_{step}"
        if not step_dir_fi.exists() or not step_dir_trt.exists():
            print(f"\n=== Step {step} ===")
            print("  Missing step directory in one of the backends, skipping")
            continue
        print(f"\n=== Step {step} ===")
        res = compare_step(
            step_dir_fi,
            step_dir_trt,
            args.threshold,
            args.kv_max_elements,
            args.kv_sample_pages,
            args.kv_compare,
        )
        res["step"] = step
        all_results.append(res)

    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved results to {args.output}")
        except Exception as e:
            print(f"Failed to save results: {e}")

    # Simple summary
    if all_results:
        worst = []
        for r in all_results:
            attn = r["tensors"].get("attn_out", {})
            worst.append((r["step"], attn.get("max_abs_diff", float("nan"))))
        worst = [x for x in worst if isinstance(x[1], float)]
        if worst:
            worst.sort(key=lambda x: (float("inf") if x[1] != x[1] else x[1]), reverse=True)
            s, v = worst[0]
            print(f"\nWorst step by attn_out max diff: step {s}: {v:.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


