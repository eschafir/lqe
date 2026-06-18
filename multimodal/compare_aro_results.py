import json
import os
import argparse

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_clip = os.path.join(project_root, "results", "mlqe_aro_results.json")
    default_vlm = os.path.join(project_root, "results", "vlm_aro_results.json")
    default_output = os.path.join(project_root, "results", "aro_comparison_results.md")

    parser = argparse.ArgumentParser(description="Compare CLIP and VLM ARO Results")
    parser.add_argument("--clip-results", type=str, default=default_clip, help="Path to CLIP results JSON")
    parser.add_argument("--vlm-results", type=str, default=default_vlm, help="Path to VLM results JSON")
    parser.add_argument("--output", type=str, default=default_output, help="Path to save comparison markdown")
    args = parser.parse_args()
    
    if not os.path.exists(args.clip_results):
        print(f"Error: CLIP results file not found at {args.clip_results}")
        print("Please run: python multimodal/multimodal_lqe_eval.py --num-samples 100 --output results/mlqe_aro_results.json")
        return
        
    if not os.path.exists(args.vlm_results):
        print(f"Error: VLM results file not found at {args.vlm_results}")
        print("Please run: python multimodal/multimodal_vlm_eval.py --provider nim --model meta/llama-3.2-11b-vision-instruct --num-samples 100 --output results/vlm_aro_results.json")
        return

    with open(args.clip_results, "r") as f:
        clip_data = json.load(f)
        
    with open(args.vlm_results, "r") as f:
        vlm_data = json.load(f)
        
    n_clip = clip_data.get("num_samples", 0)
    n_vlm = vlm_data.get("num_samples", 0)
    
    output_lines = []
    
    output_lines.append("=" * 60)
    output_lines.append("           ARO ATTRIBUTION BENCHMARK COMPARISON")
    output_lines.append("=" * 60)
    output_lines.append(f"Total Samples Evaluated: CLIP = {n_clip} | VLM = {n_vlm}")
    output_lines.append("\n### Performance Comparison Table")
    output_lines.append("| Method | Accuracy | Accuracy (%) | Description |")
    output_lines.append("| :--- | :---: | :---: | :--- |")
    output_lines.append(f"| **Vanilla CLIP** | {clip_data.get('vanilla_accuracy', 0):.4f} | {clip_data.get('vanilla_accuracy', 0)*100:.2f}% | Direct cross-modal sentence score |")
    output_lines.append(f"| **M-LQE (Average)** | {clip_data.get('mlqe_avg_accuracy', 0):.4f} | {clip_data.get('mlqe_avg_accuracy', 0)*100:.2f}% | Mean component score (templates) |")
    output_lines.append(f"| **M-LQE (Product)** | {clip_data.get('mlqe_prod_accuracy', 0):.4f} | {clip_data.get('mlqe_prod_accuracy', 0)*100:.2f}% | Product component score (templates) |")
    output_lines.append(f"| **M-LQE (Hybrid)** | {clip_data.get('mlqe_hybrid_accuracy', 0):.4f} | {clip_data.get('mlqe_hybrid_accuracy', 0)*100:.2f}% | Global score + 0.5 * component score |")
    output_lines.append(f"| **M-LQE (Grounded Crop)** | {clip_data.get('mlqe_grounded_accuracy', 0):.4f} | {clip_data.get('mlqe_grounded_accuracy', 0)*100:.2f}% | Target object bounding-box crop |")
    output_lines.append(f"| **M-LQE (Grounded Fusion)** | {clip_data.get('mlqe_fusion_accuracy', 0):.4f} | {clip_data.get('mlqe_fusion_accuracy', 0)*100:.2f}% | Global score + 0.5 * crop score |")
    output_lines.append(f"| **VLM ({vlm_data.get('model', 'Llama-3.2-Vision')})** | {vlm_data.get('accuracy', 0):.4f} | {vlm_data.get('accuracy', 0)*100:.2f}% | Cross-attention visual reasoning |")
    output_lines.append("=" * 60)
    
    # Analyze detailed examples (CLIP Failure vs VLM Success)
    clip_details = {item["id"]: item for item in clip_data.get("details", [])}
    vlm_details = {item["id"]: item for item in vlm_data.get("details", [])}
    
    output_lines.append("\n### Examples of CLIP Attribute-Binding Failures Solved by VLM")
    output_lines.append("Below are instances where Vanilla CLIP was fooled by swapped attributes, but the VLM reasoned correctly:")
    
    shown = 0
    common_ids = sorted(list(set(clip_details.keys()) & set(vlm_details.keys())))
    
    for q_id in common_ids:
        c_item = clip_details[q_id]
        v_item = vlm_details[q_id]
        
        # CLIP failed, VLM succeeded
        if c_item["vanilla_correct"] == 0 and v_item["correct"] == 1:
            shown += 1
            output_lines.append(f"\n{shown}. **Sample ID {q_id}**:")
            output_lines.append(f"   * **True Caption**: '{c_item['true_caption']}'")
            output_lines.append(f"   * **False Caption**: '{c_item['false_caption']}'")
            output_lines.append(f"   * **VLM Selected**: Option {v_item['pred_label']} (Raw Output: '{v_item['raw_output']}')")
            if shown >= 5:
                break
                
    if shown == 0:
        output_lines.append("   No matching fail-vs-success instances found in the overlapping sample range.")

    full_output = "\n".join(output_lines)
    print(full_output)
    
    if args.output:
        # Create directory if it doesn't exist
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(full_output)
        print(f"\nComparison results successfully saved to: {args.output}")

if __name__ == "__main__":
    main()
