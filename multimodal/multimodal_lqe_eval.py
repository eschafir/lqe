import os
import re
import sys
# Force torchaudio to be treated as unavailable to prevent binary loading crashes
sys.modules['torchaudio'] = None

import json
import argparse
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset

# Add project root parent to sys.path to find src.models
project_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_parent)
sys.path.insert(0, os.path.join(project_parent, "subspace-search"))

def rule_based_parse(caption: str) -> list[str]:
    """Parse a simple ARO caption into distinct object-attribute components using regex rules."""
    # Example: "the open door and the crouched man" -> ["open door", "crouched man"]
    parts = caption.split(" and ")
    components = []
    for part in parts:
        part = part.strip()
        # Remove leading articles like 'the', 'a', 'an'
        part = re.sub(r"^(the|a|an)\s+", "", part, flags=re.IGNORECASE)
        components.append(part)
    return components

def parse_with_llm(caption: str, model, tokenizer, device: str) -> list[str]:
    """Use the local LLM to extract object-attribute components from the caption."""
    prompt = (
        "Instructions: Parse the following text into distinct noun-adjective segments. "
        "Remove leading articles ('the', 'a', 'an'). Format the output as a simple comma-separated list.\n\n"
        "Examples:\n"
        "Text: 'the open door and the crouched man'\n"
        "Output: open door, crouched man\n\n"
        "Text: 'a black dog and a red ball'\n"
        "Output: black dog, red ball\n\n"
        "Text: '" + caption + "'\n"
        "Output:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    
    generated_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # Split by comma
    components = [c.strip() for c in generated_text.split(",") if c.strip()]
    if not components:
        # Fallback to rule-based if LLM output is empty or malformed
        return rule_based_parse(caption)
    return components

def extract_tensor(output):
    """Safely extract the raw tensor from CLIPModel outputs across different transformers versions."""
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (list, tuple)):
        return output[0]
    return output

def main():
    parser = argparse.ArgumentParser(description="M-LQE Evaluation on ARO Visual Attribution")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model key")
    parser.add_argument("--llm-model", type=str, default="qwen-1.5b", help="LLM key for parser (if using llm parser)")
    parser.add_argument("--parser", type=str, choices=["rule", "llm"], default="rule", help="Query parsing strategy")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of test samples to evaluate")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--output", type=str, default=os.path.join(project_root, "results", "mlqe_aro_results.json"), help="Path to save result JSON")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # 1. Load CLIP Model
    from transformers import CLIPProcessor, CLIPModel
    print(f"Loading CLIP model: {args.model}...")
    clip_model = CLIPModel.from_pretrained(args.model, use_safetensors=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.model)
    
    # 2. Optionally load LLM Model
    llm_model, tokenizer = None, None
    if args.parser == "llm":
        print(f"Loading local LLM for parsing: {args.llm_model}...")
        from src.models import load
        llm_model, tokenizer = load(args.llm_model, device=device)
        
    # 3. Load ARO Visual Attribution dataset from Hugging Face
    print("Loading ARO Visual Attribution dataset from HF...")
    dataset = load_dataset("gowitheflow/ARO-Visual-Attribution", split="test", streaming=True)
    
    vanilla_correct_count = 0
    mlqe_avg_correct_count = 0
    mlqe_prod_correct_count = 0
    mlqe_hybrid_correct_count = 0
    mlqe_grounded_correct_count = 0
    mlqe_fusion_correct_count = 0
    total_count = 0
    
    results_log = []
    
    print("\n--- Starting Evaluation ---")
    for idx, sample in enumerate(dataset):
        if total_count >= args.num_samples:
            break
            
        img = sample["image"]
        true_cap = sample["true_caption"]
        false_cap = sample["false_caption"]
        obj1 = sample.get("obj1", "")
        attributes = sample.get("attributes", [])
        
        attr_true = attributes[0] if len(attributes) > 0 else ""
        attr_false = attributes[1] if len(attributes) > 1 else ""
        
        # 3.1. Compute CLIP image embedding
        inputs_img = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_embed = clip_model.get_image_features(**inputs_img)
            img_embed = extract_tensor(img_embed)
            img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
            img_embed_np = img_embed.cpu().numpy()[0]
            
        # 3.2. Parse captions
        if args.parser == "llm":
            true_comps = parse_with_llm(true_cap, llm_model, tokenizer, device)
            false_comps = parse_with_llm(false_cap, llm_model, tokenizer, device)
        else:
            true_comps = rule_based_parse(true_cap)
            false_comps = rule_based_parse(false_cap)
            
        # 3.3. Evaluate Vanilla CLIP (matching full caption)
        inputs_t_full = clip_processor(text=[true_cap], return_tensors="pt", padding=True).to(device)
        inputs_f_full = clip_processor(text=[false_cap], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            t_full_embed = clip_model.get_text_features(**inputs_t_full)
            t_full_embed = extract_tensor(t_full_embed)
            t_full_embed = t_full_embed / t_full_embed.norm(p=2, dim=-1, keepdim=True)
            t_full_embed_np = t_full_embed.cpu().numpy()[0]
            
            f_full_embed = clip_model.get_text_features(**inputs_f_full)
            f_full_embed = extract_tensor(f_full_embed)
            f_full_embed = f_full_embed / f_full_embed.norm(p=2, dim=-1, keepdim=True)
            f_full_embed_np = f_full_embed.cpu().numpy()[0]
            
        vanilla_true_score = float(np.dot(img_embed_np, t_full_embed_np))
        vanilla_false_score = float(np.dot(img_embed_np, f_full_embed_np))
        vanilla_correct = 1 if vanilla_true_score > vanilla_false_score else 0
        vanilla_correct_count += vanilla_correct
        
        # 3.4. Evaluate M-LQE (matching decomposed components with in-distribution templates)
        t_comp_scores = []
        for comp in true_comps:
            comp_text = f"a photo of a {comp}"
            inputs_comp = clip_processor(text=[comp_text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                c_embed = clip_model.get_text_features(**inputs_comp)
                c_embed = extract_tensor(c_embed)
                c_embed = c_embed / c_embed.norm(p=2, dim=-1, keepdim=True)
                c_embed_np = c_embed.cpu().numpy()[0]
            t_comp_scores.append(float(np.dot(img_embed_np, c_embed_np)))
            
        f_comp_scores = []
        for comp in false_comps:
            comp_text = f"a photo of a {comp}"
            inputs_comp = clip_processor(text=[comp_text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                c_embed = clip_model.get_text_features(**inputs_comp)
                c_embed = extract_tensor(c_embed)
                c_embed = c_embed / c_embed.norm(p=2, dim=-1, keepdim=True)
                c_embed_np = c_embed.cpu().numpy()[0]
            f_comp_scores.append(float(np.dot(img_embed_np, c_embed_np)))
            
        # 3.5. Grounded Visual Verification (Cropping based on object bounding box)
        bbox = sample.get("bbox", None)
        has_crop = bbox is not None and all(k in bbox for k in ["x", "y", "w", "h"]) and obj1 and attr_true and attr_false
        cropped_img = None
        
        if has_crop:
            try:
                x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
                left = max(0, x)
                top = max(0, y)
                right = min(img.width, x + w)
                bottom = min(img.height, y + h)
                if right > left and bottom > top:
                    cropped_img = img.crop((left, top, right, bottom))
            except Exception:
                cropped_img = None
                
        cropped_true_score = 0.0
        cropped_false_score = 0.0
        if cropped_img is not None:
            inputs_crop = clip_processor(images=cropped_img, return_tensors="pt").to(device)
            true_crop_text = f"a photo of a {attr_true} {obj1}"
            false_crop_text = f"a photo of a {attr_false} {obj1}"
            inputs_t_crop = clip_processor(text=[true_crop_text], return_tensors="pt", padding=True).to(device)
            inputs_f_crop = clip_processor(text=[false_crop_text], return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                crop_embed = clip_model.get_image_features(**inputs_crop)
                crop_embed = extract_tensor(crop_embed)
                crop_embed = crop_embed / crop_embed.norm(p=2, dim=-1, keepdim=True)
                crop_embed_np = crop_embed.cpu().numpy()[0]
                
                t_crop_embed = clip_model.get_text_features(**inputs_t_crop)
                t_crop_embed = extract_tensor(t_crop_embed)
                t_crop_embed = t_crop_embed / t_crop_embed.norm(p=2, dim=-1, keepdim=True)
                t_crop_embed_np = t_crop_embed.cpu().numpy()[0]
                
                f_crop_embed = clip_model.get_text_features(**inputs_f_crop)
                f_crop_embed = extract_tensor(f_crop_embed)
                f_crop_embed = f_crop_embed / f_crop_embed.norm(p=2, dim=-1, keepdim=True)
                f_crop_embed_np = f_crop_embed.cpu().numpy()[0]
                
            cropped_true_score = float(np.dot(crop_embed_np, t_crop_embed_np))
            cropped_false_score = float(np.dot(crop_embed_np, f_crop_embed_np))
            
        # Scores combination (Average, Product, Hybrid, Grounded, and Grounded Fusion)
        mlqe_true_avg = np.mean(t_comp_scores) if t_comp_scores else 0.0
        mlqe_false_avg = np.mean(f_comp_scores) if f_comp_scores else 0.0
        mlqe_avg_correct = 1 if mlqe_true_avg > mlqe_false_avg else 0
        mlqe_avg_correct_count += mlqe_avg_correct
        
        mlqe_true_prod = np.prod(t_comp_scores) if t_comp_scores else 0.0
        mlqe_false_prod = np.prod(f_comp_scores) if f_comp_scores else 0.0
        mlqe_prod_correct = 1 if mlqe_true_prod > mlqe_false_prod else 0
        mlqe_prod_correct_count += mlqe_prod_correct
        
        beta = 0.5
        mlqe_true_hybrid = vanilla_true_score + beta * mlqe_true_avg
        mlqe_false_hybrid = vanilla_false_score + beta * mlqe_false_avg
        mlqe_hybrid_correct = 1 if mlqe_true_hybrid > mlqe_false_hybrid else 0
        mlqe_hybrid_correct_count += mlqe_hybrid_correct
        
        # Grounded (Crop-Only) Score
        mlqe_true_grounded = cropped_true_score
        mlqe_false_grounded = cropped_false_score
        mlqe_grounded_correct = 1 if mlqe_true_grounded > mlqe_false_grounded else 0
        if cropped_img is not None:
            mlqe_grounded_correct_count += mlqe_grounded_correct
        
        # Grounded Fusion Score: global score + gamma * grounded crop score
        gamma = 0.5
        mlqe_true_fusion = vanilla_true_score + gamma * cropped_true_score
        mlqe_false_fusion = vanilla_false_score + gamma * cropped_false_score
        mlqe_fusion_correct = 1 if mlqe_true_fusion > mlqe_false_fusion else 0
        if cropped_img is not None:
            mlqe_fusion_correct_count += mlqe_fusion_correct
        else:
            # Fallback to vanilla if crop is missing
            mlqe_fusion_correct = vanilla_correct
            mlqe_fusion_correct_count += vanilla_correct
        
        total_count += 1
        print(f"Sample {total_count:02d} | True: '{true_cap}' vs False: '{false_cap}'")
        print(f"  Vanilla:      {vanilla_true_score:.4f} vs {vanilla_false_score:.4f} | Correct: {vanilla_correct}")
        print(f"  M-LQE (Avg):  {mlqe_true_avg:.4f} vs {mlqe_false_avg:.4f} | Correct: {mlqe_avg_correct}")
        print(f"  M-LQE (Prod): {mlqe_true_prod:.6f} vs {mlqe_false_prod:.6f} | Correct: {mlqe_prod_correct}")
        print(f"  M-LQE (Hyb):  {mlqe_true_hybrid:.4f} vs {mlqe_false_hybrid:.4f} | Correct: {mlqe_hybrid_correct}")
        if cropped_img is not None:
            print(f"  M-LQE (Grd):  {mlqe_true_grounded:.4f} vs {mlqe_false_grounded:.4f} | Correct: {mlqe_grounded_correct}")
            print(f"  M-LQE (Fus):  {mlqe_true_fusion:.4f} vs {mlqe_false_fusion:.4f} | Correct: {mlqe_fusion_correct}")
        
        results_log.append({
            "id": total_count,
            "true_caption": true_cap,
            "false_caption": false_cap,
            "vanilla_correct": vanilla_correct,
            "mlqe_avg_correct": mlqe_avg_correct,
            "mlqe_prod_correct": mlqe_prod_correct,
            "mlqe_hybrid_correct": mlqe_hybrid_correct,
            "mlqe_grounded_correct": mlqe_grounded_correct if cropped_img is not None else None,
            "mlqe_fusion_correct": mlqe_fusion_correct
        })
        
    vanilla_acc = vanilla_correct_count / total_count
    mlqe_avg_acc = mlqe_avg_correct_count / total_count
    mlqe_prod_acc = mlqe_prod_correct_count / total_count
    mlqe_hybrid_acc = mlqe_hybrid_correct_count / total_count
    
    # Calculate crop-based accuracy over samples that actually had valid crops
    valid_crop_count = sum(1 for r in results_log if r["mlqe_grounded_correct"] is not None)
    mlqe_grd_acc = (mlqe_grounded_correct_count / valid_crop_count) if valid_crop_count > 0 else 0.0
    mlqe_fus_acc = mlqe_fusion_correct_count / total_count
    
    print("\n" + "=" * 50)
    print("M-LQE Evaluation Summary on ARO Attribution")
    print("=" * 50)
    print(f"Total Samples Evaluated: {total_count}")
    print(f"Vanilla CLIP Accuracy:       {vanilla_acc:.2%}")
    print(f"M-LQE (Average Similarity): {mlqe_avg_acc:.2%}")
    print(f"M-LQE (Product Similarity): {mlqe_prod_acc:.2%}")
    print(f"M-LQE (Hybrid Fusion Score): {mlqe_hybrid_acc:.2%}")
    print(f"M-LQE (Grounded Crop-Only): {mlqe_grd_acc:.2%} (over {valid_crop_count} valid crops)")
    print(f"M-LQE (Grounded Fusion):    {mlqe_fus_acc:.2%}")
    print("=" * 50)
    
    # Save results to JSON
    summary = {
        "dataset": "gowitheflow/ARO-Visual-Attribution",
        "num_samples": total_count,
        "vanilla_accuracy": vanilla_acc,
        "mlqe_avg_accuracy": mlqe_avg_acc,
        "mlqe_prod_accuracy": mlqe_prod_acc,
        "mlqe_hybrid_accuracy": mlqe_hybrid_acc,
        "mlqe_grounded_accuracy": mlqe_grd_acc,
        "mlqe_fusion_accuracy": mlqe_fus_acc,
        "details": results_log
    }
    
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")
    
    # Bypass PyGILState_Release finalization crash
    os._exit(0)

if __name__ == "__main__":
    main()
