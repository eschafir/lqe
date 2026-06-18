import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compare CLIP and VLM Winoground Results")
    parser.add_argument("--clip-results", type=str, default="winoground_clip_results.json", help="Path to CLIP results JSON")
    parser.add_argument("--vlm-results", type=str, default="winoground_vlm_results.json", help="Path to VLM results JSON")
    parser.add_argument("--output", type=str, default="", help="Path to save comparison markdown")
    args = parser.parse_args()
    
    if not os.path.exists(args.clip_results):
        print(f"Error: CLIP results file not found at {args.clip_results}")
        print("Please run: python winoground_lqe_eval.py --num-samples 100 --output winoground_clip_results.json")
        return
        
    if not os.path.exists(args.vlm_results):
        print(f"Error: VLM results file not found at {args.vlm_results}")
        print("Please run: python winoground_vlm_eval.py --provider nim --model meta/llama-3.2-11b-vision-instruct --num-samples 100 --output winoground_vlm_results.json")
        return

    with open(args.clip_results, "r") as f:
        clip_data = json.load(f)
        
    with open(args.vlm_results, "r") as f:
        vlm_data = json.load(f)
        
    n_clip = clip_data.get("num_samples", 0)
    n_vlm = vlm_data.get("num_samples", 0)
    
    clip_res = clip_data.get("results", {})
    vlm_res = vlm_data.get("results", {})
    
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("           WINOGROUND COMPOSITIONALITY BENCHMARK COMPARISON")
    output_lines.append("=" * 70)
    output_lines.append(f"Total Samples Evaluated: CLIP = {n_clip} | VLM = {n_vlm}")
    output_lines.append("\n### Winoground Performance Comparison Table")
    output_lines.append("| Method | Text Score (%) | Image Score (%) | Group Score (%) | Description |")
    output_lines.append("| :--- | :---: | :---: | :---: | :--- |")
    
    # 1. Vanilla CLIP
    van = clip_res.get("vanilla", {})
    output_lines.append(
        f"| **Vanilla CLIP** | {van.get('text_accuracy', 0)*100:.2f}% | {van.get('image_accuracy', 0)*100:.2f}% | {van.get('group_accuracy', 0)*100:.2f}% | Direct cross-modal score |"
    )
    # 2. M-LQE Average
    avg = clip_res.get("mlqe_avg", {})
    output_lines.append(
        f"| **M-LQE (Average)** | {avg.get('text_accuracy', 0)*100:.2f}% | {avg.get('image_accuracy', 0)*100:.2f}% | {avg.get('group_accuracy', 0)*100:.2f}% | Mean component score |"
    )
    # 3. M-LQE Product
    prod = clip_res.get("mlqe_prod", {})
    output_lines.append(
        f"| **M-LQE (Product)** | {prod.get('text_accuracy', 0)*100:.2f}% | {prod.get('image_accuracy', 0)*100:.2f}% | {prod.get('group_accuracy', 0)*100:.2f}% | Product component score |"
    )
    # 4. M-LQE Hybrid
    hyb = clip_res.get("mlqe_hybrid", {})
    output_lines.append(
        f"| **M-LQE (Hybrid)** | {hyb.get('text_accuracy', 0)*100:.2f}% | {hyb.get('image_accuracy', 0)*100:.2f}% | {hyb.get('group_accuracy', 0)*100:.2f}% | Global + 0.5 * component |"
    )
    # 5. VLM
    output_lines.append(
        f"| **VLM ({vlm_data.get('model', 'Llama-3.2-Vision')})** | {vlm_res.get('text_accuracy', 0)*100:.2f}% | {vlm_res.get('image_accuracy', 0)*100:.2f}% | {vlm_res.get('group_accuracy', 0)*100:.2f}% | Cross-attention reasoning |"
    )
    output_lines.append("=" * 70)
    
    # Showcase some failure/success examples
    clip_details = {item["id"]: item for item in clip_data.get("details", [])}
    vlm_details = {item["id"]: item for item in vlm_data.get("details", [])}
    
    output_lines.append("\n### Examples of CLIP Failures Solved by VLM on Winoground")
    output_lines.append("Below are samples where Vanilla CLIP failed (Group Score = 0), but the VLM succeeded (Group Score = 1):")
    
    shown = 0
    common_ids = sorted(list(set(clip_details.keys()) & set(vlm_details.keys())))
    
    for q_id in common_ids:
        c_item = clip_details[q_id]
        v_item = vlm_details[q_id]
        
        # CLIP failed on group, VLM succeeded on group
        if c_item["vanilla_group"] == 0 and v_item["group_correct"] == 1:
            shown += 1
            output_lines.append(f"\n{shown}. **Sample ID {q_id}**:")
            output_lines.append(f"   * **Caption 0**: '{c_item['caption_0']}'")
            output_lines.append(f"   * **Caption 1**: '{c_item['caption_1']}'")
            output_lines.append(f"   * **CLIP Vanilla Score**: Text Correct = {c_item['vanilla_text']} | Image Correct = {c_item['vanilla_image']}")
            output_lines.append(f"   * **VLM Correctness**: Text Correct = {v_item['text_correct']} | Image Correct = {v_item['image_correct']}")
            if shown >= 5:
                break
                
    if shown == 0:
        output_lines.append("   No matching instances found in the overlapping sample range.")

    full_output = "\n".join(output_lines)
    print(full_output)
    
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(full_output)
        print(f"\nWinoground comparison results successfully saved to: {args.output}")

if __name__ == "__main__":
    main()
