import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compare CLIP and VLM ARO Results")
    parser.add_argument("--clip-results", type=str, default="clip_results.json", help="Path to CLIP results JSON")
    parser.add_argument("--vlm-results", type=str, default="vlm_results.json", help="Path to VLM results JSON")
    args = parser.parse_args()
    
    if not os.path.exists(args.clip_results):
        print(f"Error: CLIP results file not found at {args.clip_results}")
        print("Please run: python multimodal_lqe_eval.py --num-samples 100 --output clip_results.json")
        return
        
    if not os.path.exists(args.vlm_results):
        print(f"Error: VLM results file not found at {args.vlm_results}")
        print("Please run: python multimodal_vlm_eval.py --provider nim --base-url https://integrate.api.nvidia.com/v1 --model meta/llama-3.2-11b-vision-instruct --num-samples 100 --output vlm_results.json")
        return

    with open(args.clip_results, "r") as f:
        clip_data = json.load(f)
        
    with open(args.vlm_results, "r") as f:
        vlm_data = json.load(f)
        
    n_clip = clip_data.get("num_samples", 0)
    n_vlm = vlm_data.get("num_samples", 0)
    
    print("\n" + "=" * 60)
    print("           ARO ATTRIBUTION BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"Total Samples Evaluated: CLIP = {n_clip} | VLM = {n_vlm}")
    print("\n### Performance Comparison Table")
    print("| Method | Accuracy | Accuracy (%) | Description |")
    print("| :--- | :---: | :---: | :--- |")
    print(f"| **Vanilla CLIP** | {clip_data.get('vanilla_accuracy', 0):.4f} | {clip_data.get('vanilla_accuracy', 0)*100:.2%}| Direct cross-modal sentence score |")
    print(f"| **M-LQE (Average)** | {clip_data.get('mlqe_avg_accuracy', 0):.4f} | {clip_data.get('mlqe_avg_accuracy', 0)*100:.2%} | Mean component score (templates) |")
    print(f"| **M-LQE (Product)** | {clip_data.get('mlqe_prod_accuracy', 0):.4f} | {clip_data.get('mlqe_prod_accuracy', 0)*100:.2%} | Product component score (templates) |")
    print(f"| **M-LQE (Hybrid)** | {clip_data.get('mlqe_hybrid_accuracy', 0):.4f} | {clip_data.get('mlqe_hybrid_accuracy', 0)*100:.2%} | Global score + 0.5 * component score |")
    print(f"| **M-LQE (Grounded Crop)** | {clip_data.get('mlqe_grounded_accuracy', 0):.4f} | {clip_data.get('mlqe_grounded_accuracy', 0)*100:.2%} | Target object bounding-box crop |")
    print(f"| **M-LQE (Grounded Fusion)** | {clip_data.get('mlqe_fusion_accuracy', 0):.4f} | {clip_data.get('mlqe_fusion_accuracy', 0)*100:.2%} | Global score + 0.5 * crop score |")
    print(f"| **VLM ({vlm_data.get('model', 'Llama-3.2-Vision')})** | {vlm_data.get('accuracy', 0):.4f} | {vlm_data.get('accuracy', 0)*100:.2%} | Cross-attention visual reasoning |")
    print("=" * 60)
    
    # Analyze detailed examples (CLIP Failure vs VLM Success)
    clip_details = {item["id"]: item for item in clip_data.get("details", [])}
    vlm_details = {item["id"]: item for item in vlm_data.get("details", [])}
    
    print("\n### Examples of CLIP Attribute-Binding Failures Solved by VLM")
    print("Below are instances where Vanilla CLIP was fooled by swapped attributes, but the VLM reasoned correctly:")
    
    shown = 0
    common_ids = sorted(list(set(clip_details.keys()) & set(vlm_details.keys())))
    
    for q_id in common_ids:
        c_item = clip_details[q_id]
        v_item = vlm_details[q_id]
        
        # CLIP failed, VLM succeeded
        if c_item["vanilla_correct"] == 0 and v_item["correct"] == 1:
            shown += 1
            print(f"\n{shown}. **Sample ID {q_id}**:")
            print(f"   * **True Caption**: '{c_item['true_caption']}'")
            print(f"   * **False Caption**: '{c_item['false_caption']}'")
            print(f"   * **VLM Selected**: Option {v_item['pred_label']} (Raw Output: '{v_item['raw_output']}')")
            if shown >= 5:
                break
                
    if shown == 0:
        print("   No matching fail-vs-success instances found in the overlapping sample range.")

if __name__ == "__main__":
    main()
