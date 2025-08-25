#!/usr/bin/env python3
"""
Compare MLA attention tensors between flashinfer and trtllm backends.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
import json

def load_tensor_safely(filepath):
    """Load tensor with error handling."""
    try:
        if not os.path.exists(filepath):
            return None
        return torch.load(filepath, map_location='cpu')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compute_tensor_diff_stats(tensor1, tensor2, name="tensor"):
    """Compute comprehensive difference statistics between two tensors."""
    if tensor1 is None or tensor2 is None:
        return {"error": "One or both tensors are None"}
    
    if tensor1.shape != tensor2.shape:
        return {"error": f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"}
    
    # Convert to float32 for more precise calculations
    t1 = tensor1.float()
    t2 = tensor2.float()
    
    # Compute differences
    abs_diff = torch.abs(t1 - t2)
    rel_diff = abs_diff / (torch.abs(t1) + 1e-8)
    
    stats = {
        "shape": list(tensor1.shape),
        "dtype": str(tensor1.dtype),
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "t1_norm": torch.norm(t1).item(),
        "t2_norm": torch.norm(t2).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            t1.flatten(), t2.flatten(), dim=0
        ).item(),
        "num_elements": tensor1.numel(),
        "large_diff_count": (abs_diff > 1e-3).sum().item(),
        "large_diff_percentage": ((abs_diff > 1e-3).sum().item() / tensor1.numel()) * 100,
    }
    
    # Add percentile information
    abs_diff_flat = abs_diff.flatten()
    percentiles = [50, 90, 95, 99, 99.9]
    for p in percentiles:
        stats[f"abs_diff_p{p}"] = torch.quantile(abs_diff_flat, p/100.0).item()
    
    return stats

def print_tensor_head(t, name, n=10):
    """Print the first n values of a tensor, flattened."""
    t_flat = t.flatten()
    vals = t_flat[:n].tolist()
    print(f"    {name}[:{n}]: {vals}")

def compare_step_tensors(step_dir1, step_dir2, step_num):
    """Compare all tensors for a given step between two backends."""
    tensor_files = [
        "hidden_states_at_entry.pt",  # Hidden states at entry
        "q_after_layernorm.pt",  # Q after RMSNorm (LoRA path)
        "q_input_b_proj.pt",     # Input to q_b_proj (LoRA path)
        "q_rope_pre.pt",  # Q before RoPE rotation (debug)
        "q_rope.pt",      # Q after RoPE rotation
        "q_nope_raw.pt",  # Q NoPE before BMM (debug)
        "q_nope.pt",      # Q NoPE after BMM
        "q_lora_final.pt", # Q after q_b_proj (LoRA path)
        "q_b_proj_weight.pt", # q_b_proj weight tensor
        "k_nope.pt", 
        "k_rope.pt", 
        "attn_out.pt"
    ]
    
    step_results = {
        "step": step_num,
        "tensors": {},
        "summary": {}
    }
    
    print(f"\n=== Step {step_num} ===")
    
    # Load and compare meta info
    meta1 = load_tensor_safely(step_dir1 / "meta.pt")
    meta2 = load_tensor_safely(step_dir2 / "meta.pt")
    
    if meta1 and meta2:
        print(f"Meta - Layer: {meta1.get('layer_id', 'N/A')}, "
              f"Seq lens: {meta1.get('seq_lens', 'N/A')}")
    
    total_max_diff = 0
    total_mean_diff = 0
    tensor_count = 0
    
    # Tensor descriptions for better output
    tensor_descriptions = {
        "hidden_states_at_entry": "Hidden States at Entry",
        "q_after_layernorm": "Q After Layernorm",
        "q_input_b_proj": "Q Input B Proj",
        "q_rope_pre": "Q RoPE (pre-rotation)",
        "q_rope": "Q RoPE (post-rotation)", 
        "q_nope_raw": "Q NoPE (pre-BMM)",
        "q_nope": "Q NoPE (post-BMM)",
        "q_lora_final": "Q LoRA Final",
        "q_b_proj_weight": "Q B Proj Weight",
        "k_nope": "K NoPE",
        "k_rope": "K RoPE",
        "attn_out": "Attention Output"
    }
    
    for tensor_file in tensor_files:
        tensor_name = tensor_file.replace('.pt', '')
        tensor_desc = tensor_descriptions.get(tensor_name, tensor_name)
        
        t1 = load_tensor_safely(step_dir1 / tensor_file)
        t2 = load_tensor_safely(step_dir2 / tensor_file)
        
        if t1 is None or t2 is None:
            print(f"  {tensor_desc:20s}: MISSING")
            step_results["tensors"][tensor_name] = {"error": "Missing tensor"}
            continue
        
        stats = compute_tensor_diff_stats(t1, t2, tensor_name)
        step_results["tensors"][tensor_name] = stats
        
        if "error" in stats:
            print(f"  {tensor_desc:20s}: ERROR - {stats['error']}")
            continue
        
        # Update totals
        total_max_diff = max(total_max_diff, stats["max_abs_diff"])
        total_mean_diff += stats["mean_abs_diff"]
        tensor_count += 1
        
        # Print summary
        print(f"  {tensor_desc:20s}: "
              f"max_diff={stats['max_abs_diff']:.2e}, "
              f"mean_diff={stats['mean_abs_diff']:.2e}, "
              f"cos_sim={stats['cosine_similarity']:.6f}, "
              f"large_diff={stats['large_diff_percentage']:.1f}%")
        
        # Print first 10 values of each tensor and their diff
        print_tensor_head(t1, f"{tensor_name} (backend1)")
        print_tensor_head(t2, f"{tensor_name} (backend2)")
        print_tensor_head(torch.abs(t1.float() - t2.float()), f"{tensor_name} (abs_diff)")
        
        # Flag concerning differences
        if stats["max_abs_diff"] > 1e-2:
            print(f"    ⚠️  LARGE MAX DIFFERENCE: {stats['max_abs_diff']:.2e}")
        if stats["cosine_similarity"] < 0.99:
            print(f"    ⚠️  LOW COSINE SIMILARITY: {stats['cosine_similarity']:.6f}")
        if stats["large_diff_percentage"] > 10:
            print(f"    ⚠️  MANY LARGE DIFFS: {stats['large_diff_percentage']:.1f}%")
    
    step_results["summary"] = {
        "max_abs_diff_across_tensors": total_max_diff,
        "avg_mean_diff": total_mean_diff / max(tensor_count, 1),
        "tensor_count": tensor_count
    }
    
    return step_results

def generate_summary_report(all_results, output_file=None):
    """Generate a summary report of all comparisons."""
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    
    # Overall statistics
    worst_steps = []
    tensor_trends = {}
    
    for result in all_results:
        step = result["step"]
        max_diff = result["summary"]["max_abs_diff_across_tensors"]
        worst_steps.append((step, max_diff))
        
        # Track per-tensor trends
        for tensor_name, stats in result["tensors"].items():
            if "error" not in stats:
                if tensor_name not in tensor_trends:
                    tensor_trends[tensor_name] = []
                tensor_trends[tensor_name].append((step, stats["max_abs_diff"]))
    
    # Sort worst steps
    worst_steps.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nWorst Steps (by maximum absolute difference):")
    for i, (step, max_diff) in enumerate(worst_steps[:5]):
        print(f"  {i+1}. Step {step}: {max_diff:.2e}")
    
    print(f"\nTensor Trends (max difference across steps):")
    for tensor_name, trends in tensor_trends.items():
        max_diff_step = max(trends, key=lambda x: x[1])
        min_diff_step = min(trends, key=lambda x: x[1])
        avg_diff = sum(x[1] for x in trends) / len(trends)
        
        print(f"  {tensor_name:12s}: "
              f"max={max_diff_step[1]:.2e} (step {max_diff_step[0]}), "
              f"min={min_diff_step[1]:.2e} (step {min_diff_step[0]}), "
              f"avg={avg_diff:.2e}")
    
    # Generate recommendations
    print(f"\nRecommendations:")
    
    critical_steps = [s for s, d in worst_steps if d > 1e-2]
    if critical_steps:
        print(f"  - Investigate steps with large differences: {critical_steps}")
    
    worst_tensors = []
    q_chain_issues = []
    for tensor_name, trends in tensor_trends.items():
        max_diff = max(x[1] for x in trends)
        if max_diff > 1e-2:
            worst_tensors.append(tensor_name)
            # Categorize Q processing chain issues
            if tensor_name == "q_rope_pre":
                q_chain_issues.append("Q projection/layernorm differs before RoPE")
            elif tensor_name == "q_rope" and "q_rope_pre" not in worst_tensors:
                q_chain_issues.append("RoPE application differs (positions/implementation)")
            elif tensor_name == "q_nope_raw":
                q_chain_issues.append("Q NoPE differs before BMM")
            elif tensor_name == "q_nope" and "q_nope_raw" not in worst_tensors:
                q_chain_issues.append("BMM with w_kc differs (weights/FP8 scaling)")
    
    if worst_tensors:
        print(f"  - Focus on tensors with issues: {worst_tensors}")
    
    if q_chain_issues:
        print(f"  - Q processing chain analysis:")
        for issue in q_chain_issues:
            print(f"    • {issue}")
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  - Detailed results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare MLA attention tensors between flashinfer and trtllm backends"
    )
    parser.add_argument(
        "--dump-dir", 
        type=str, 
        default="divergence_debug",
        help="Base directory containing tensor dumps"
    )
    parser.add_argument(
        "--steps", 
        type=str, 
        default="1-10",
        help="Steps to compare (e.g., '1-10' or '1,3,5')"
    )
    parser.add_argument(
        "--backend1", 
        type=str, 
        default="flashinfer",
        help="First backend name"
    )
    parser.add_argument(
        "--backend2", 
        type=str, 
        default="trtllm_mla",
        help="Second backend name"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=1e-4,
        help="Threshold for flagging large differences"
    )
    
    args = parser.parse_args()
    
    dump_dir = Path(args.dump_dir)
    backend1_dir = dump_dir / args.backend1
    backend2_dir = dump_dir / args.backend2
    
    # Parse steps
    if '-' in args.steps:
        start, end = map(int, args.steps.split('-'))
        steps = list(range(start, end + 1))
    else:
        steps = [int(s.strip()) for s in args.steps.split(',')]
    
    print(f"Comparing {args.backend1} vs {args.backend2}")
    print(f"Steps: {steps}")
    print(f"Dump directory: {dump_dir}")
    
    # Check directories exist
    if not backend1_dir.exists():
        print(f"Error: Directory {backend1_dir} does not exist")
        return 1
    if not backend2_dir.exists():
        print(f"Error: Directory {backend2_dir} does not exist")
        return 1
    
    all_results = []
    
    for step in steps:
        step1_dir = backend1_dir / f"step_{step}"
        step2_dir = backend2_dir / f"step_{step}"
        
        if not step1_dir.exists():
            print(f"Warning: {step1_dir} does not exist, skipping step {step}")
            continue
        if not step2_dir.exists():
            print(f"Warning: {step2_dir} does not exist, skipping step {step}")
            continue
        
        result = compare_step_tensors(step1_dir, step2_dir, step)
        all_results.append(result)
    
    if all_results:
        generate_summary_report(all_results, args.output)
    else:
        print("No valid comparisons found!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())