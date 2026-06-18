import os
import sys
import subprocess
import argparse

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    default_clip = os.path.join(project_root, "results", "winoground_clip_results.json")
    default_vlm = os.path.join(project_root, "results", "winoground_vlm_results.json")
    default_output = os.path.join(project_root, "results", "winoground_comparison_results.md")

    parser = argparse.ArgumentParser(
        description="Orchestrator to run both Vanilla CLIP (M-LQE) and VLM (NIM/HF) on Winoground and compare results."
    )
    parser.add_argument("--num-samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model key")
    parser.add_argument("--vlm-provider", type=str, choices=["hf", "nim"], default="nim", help="Model API provider for VLM")
    parser.add_argument("--vlm-model", type=str, default="meta/llama-3.2-11b-vision-instruct", help="VLM model name")
    parser.add_argument("--api-key", type=str, default="", help="NVIDIA NIM API key (optional, falls back to env)")
    parser.add_argument("--clip-output", type=str, default=default_clip, help="Path to save CLIP results JSON")
    parser.add_argument("--vlm-output", type=str, default=default_vlm, help="Path to save VLM results JSON")
    parser.add_argument("--output", type=str, default=default_output, help="Path to save comparison markdown table")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP evaluation step")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM evaluation step")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("           WINOGROUND MULTIMODAL EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Number of Samples : {args.num_samples}")
    print(f"CLIP Model        : {args.clip_model}")
    print(f"VLM Provider      : {args.vlm_provider.upper()}")
    print(f"VLM Model         : {args.vlm_model}")
    print(f"Comparison Output : {args.output}")
    print(f"Skip CLIP Step    : {args.skip_clip}")
    print(f"Skip VLM Step     : {args.skip_vlm}")
    print("-" * 70)
    
    # Absolute paths to python scripts
    clip_script = os.path.join(script_dir, "winoground_lqe_eval.py")
    vlm_script = os.path.join(script_dir, "winoground_vlm_eval.py")
    compare_script = os.path.join(script_dir, "winoground_compare.py")

    # 1. Run CLIP (M-LQE) Evaluation
    if not args.skip_clip:
        print("\n[Step 1/3] Running CLIP/M-LQE Winoground Evaluation...")
        clip_cmd = [
            sys.executable, clip_script,
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
    else:
        print("\n[Step 1/3] Skipping CLIP/M-LQE Winoground Evaluation (using existing results).")
        
    # 2. Run VLM Evaluation
    if not args.skip_vlm:
        print("\n[Step 2/3] Running VLM/NIM Winoground Evaluation...")
        vlm_cmd = [
            sys.executable, vlm_script,
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
    else:
        print("\n[Step 2/3] Skipping VLM/NIM Winoground Evaluation (using existing results).")
        
    # 3. Compare Results
    print("\n[Step 3/3] Comparing Winoground Results...")
    compare_cmd = [
        sys.executable, compare_script,
        "--clip-results", args.clip_output,
        "--vlm-results", args.vlm_output
    ]
    if args.output:
        compare_cmd.extend(["--output", args.output])
    print(f"Running command: {' '.join(compare_cmd)}")
    try:
        subprocess.run(compare_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running comparison script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
