import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator to run both Vanilla CLIP (M-LQE) and VLM (NIM/HF) on ARO Visual Attribution and compare results."
    )
    parser.add_argument("--num-samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model key")
    parser.add_argument("--vlm-provider", type=str, choices=["hf", "nim"], default="nim", help="Model API provider for VLM")
    parser.add_argument("--vlm-model", type=str, default="meta/llama-3.2-11b-vision-instruct", help="VLM model name")
    parser.add_argument("--api-key", type=str, default="", help="NVIDIA NIM API key (optional, falls back to env)")
    parser.add_argument("--clip-output", type=str, default="clip_results.json", help="Path to save CLIP results JSON")
    parser.add_argument("--vlm-output", type=str, default="vlm_results.json", help="Path to save VLM results JSON")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("           MULTIMODAL EVALUATION ORCHESTRATOR (CLIP vs VLM)")
    print("=" * 70)
    print(f"Number of Samples: {args.num_samples}")
    print(f"CLIP Model       : {args.clip_model}")
    print(f"VLM Provider     : {args.vlm_provider.upper()}")
    print(f"VLM Model        : {args.vlm_model}")
    print("-" * 70)
    
    # 1. Run Vanilla CLIP (M-LQE) Evaluation
    print("\n[Step 1/3] Running CLIP/M-LQE Evaluation...")
    clip_cmd = [
        sys.executable, "multimodal_lqe_eval.py",
        "--model", args.clip_model,
        "--num-samples", str(args.num_samples),
        "--output", args.clip_output
    ]
    print(f"Running command: {' '.join(clip_cmd)}")
    try:
        subprocess.run(clip_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running CLIP evaluation: {e}")
        sys.exit(1)
        
    # 2. Run VLM Evaluation
    print("\n[Step 2/3] Running VLM/NIM Evaluation...")
    vlm_cmd = [
        sys.executable, "multimodal_vlm_eval.py",
        "--provider", args.vlm_provider,
        "--model", args.vlm_model,
        "--num-samples", str(args.num_samples),
        "--output", args.vlm_output
    ]
    if args.api_key:
        vlm_cmd.extend(["--api-key", args.api_key])
    print(f"Running command: {' '.join(vlm_cmd)}")
    try:
        subprocess.run(vlm_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running VLM evaluation: {e}")
        sys.exit(1)
        
    # 3. Compare Results
    print("\n[Step 3/3] Comparing Results...")
    compare_cmd = [
        sys.executable, "compare_aro_results.py",
        "--clip-results", args.clip_output,
        "--vlm-results", args.vlm_output
    ]
    print(f"Running command: {' '.join(compare_cmd)}")
    try:
        subprocess.run(compare_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running comparison script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
