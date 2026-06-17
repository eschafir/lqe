import os
import re
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset

# Add project root and subspace-search to sys.path to find src.models
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "subspace-search"))

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
    parser.add_argument("--output", type=str, default="mlqe_aro_results.json", help="Path to save result JSON")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # 1. Load CLIP Model
    from transformers import CLIPProcessor, CLIPModel
    print(f"Loading CLIP model: {args.model}...")
    clip_model = CLIPModel.from_pretrained(args.model).to(device)
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
    total_count = 0
    
    results_log = []
    
    print("\n--- Starting Evaluation ---")
    for idx, sample in enumerate(dataset):
        if total_count >= args.num_samples:
            break
            
        img = sample["image"]
        true_cap = sample["true_caption"]
        false_cap = sample["false_caption"]
        
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
        # Compute true components similarities
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
            
        # Compute false components similarities
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
            
        # Scores combination (Average, Product, and Local-Global Hybrid)
        mlqe_true_avg = np.mean(t_comp_scores) if t_comp_scores else 0.0
        mlqe_false_avg = np.mean(f_comp_scores) if f_comp_scores else 0.0
        mlqe_avg_correct = 1 if mlqe_true_avg > mlqe_false_avg else 0
        mlqe_avg_correct_count += mlqe_avg_correct
        
        mlqe_true_prod = np.prod(t_comp_scores) if t_comp_scores else 0.0
        mlqe_false_prod = np.prod(f_comp_scores) if f_comp_scores else 0.0
        mlqe_prod_correct = 1 if mlqe_true_prod > mlqe_false_prod else 0
        mlqe_prod_correct_count += mlqe_prod_correct
        
        # Local-Global Hybrid: vanilla global score + beta * component-average score
        beta = 0.5
        mlqe_true_hybrid = vanilla_true_score + beta * mlqe_true_avg
        mlqe_false_hybrid = vanilla_false_score + beta * mlqe_false_avg
        mlqe_hybrid_correct = 1 if mlqe_true_hybrid > mlqe_false_hybrid else 0
        mlqe_hybrid_correct_count += mlqe_hybrid_correct
        
        total_count += 1
        print(f"Sample {total_count:02d} | True: '{true_cap}' vs False: '{false_cap}'")
        print(f"  Vanilla:      {vanilla_true_score:.4f} vs {vanilla_false_score:.4f} | Correct: {vanilla_correct}")
        print(f"  M-LQE (Avg):  {mlqe_true_avg:.4f} vs {mlqe_false_avg:.4f} | Correct: {mlqe_avg_correct}")
        print(f"  M-LQE (Prod): {mlqe_true_prod:.6f} vs {mlqe_false_prod:.6f} | Correct: {mlqe_prod_correct}")
        print(f"  M-LQE (Hyb):  {mlqe_true_hybrid:.4f} vs {mlqe_false_hybrid:.4f} | Correct: {mlqe_hybrid_correct}")
        
        results_log.append({
            "id": total_count,
            "true_caption": true_cap,
            "false_caption": false_cap,
            "vanilla_correct": vanilla_correct,
            "mlqe_avg_correct": mlqe_avg_correct,
            "mlqe_prod_correct": mlqe_prod_correct,
            "mlqe_hybrid_correct": mlqe_hybrid_correct
        })
        
    vanilla_acc = vanilla_correct_count / total_count
    mlqe_avg_acc = mlqe_avg_correct_count / total_count
    mlqe_prod_acc = mlqe_prod_correct_count / total_count
    mlqe_hybrid_acc = mlqe_hybrid_correct_count / total_count
    
    print("\n" + "=" * 50)
    print("M-LQE Evaluation Summary on ARO Attribution")
    print("=" * 50)
    print(f"Total Samples Evaluated: {total_count}")
    print(f"Vanilla CLIP Accuracy:      {vanilla_acc:.2%}")
    print(f"M-LQE (Average Similarity): {mlqe_avg_acc:.2%}")
    print(f"M-LQE (Product Similarity): {mlqe_prod_acc:.2%}")
    print(f"M-LQE (Hybrid Fusion Score): {mlqe_hybrid_acc:.2%}")
    print("=" * 50)
    
    # Save results to JSON
    summary = {
        "dataset": "gowitheflow/ARO-Visual-Attribution",
        "num_samples": total_count,
        "vanilla_accuracy": vanilla_acc,
        "mlqe_avg_accuracy": mlqe_avg_acc,
        "mlqe_prod_accuracy": mlqe_prod_acc,
        "mlqe_hybrid_accuracy": mlqe_hybrid_acc,
        "details": results_log
    }
    
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
