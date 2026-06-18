import os
import sys
# Force torchaudio to be treated as unavailable to prevent binary loading crashes
sys.modules['torchaudio'] = None

import re
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset

def load_env_file(dotenv_path=".env"):
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val

def parse_winoground_caption(caption: str) -> list[str]:
    # Split by common prepositions and conjunctions to isolate noun chunks
    delimiters = r"\b(and|in|on|under|next to|near|behind|in front of|at|with|over|above|below|through|inside|into)\b"
    parts = re.split(delimiters, caption, flags=re.IGNORECASE)
    components = []
    for p in parts:
        p = p.strip()
        # Remove leading articles and pronouns
        p = re.sub(r"\b(the|a|an|some|any)\b\s*", "", p, flags=re.IGNORECASE)
        # Remove trailing prepositions/punctuation
        p = re.sub(r"[^\w\s]", "", p).strip()
        if p and p not in ["and", "in", "on", "under", "next to", "near", "behind", "in front of", "at", "with", "over", "above", "below", "through", "inside", "into"]:
            components.append(p)
    return components

def extract_tensor(output):
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
    load_env_file()
    parser = argparse.ArgumentParser(description="M-LQE CLIP Evaluation on Winoground")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model key")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples to evaluate (max 400)")
    parser.add_argument("--output", type=str, default="winoground_clip_results.json", help="Path to save result JSON")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    from transformers import CLIPProcessor, CLIPModel
    print(f"Loading CLIP model: {args.model}...")
    clip_model = CLIPModel.from_pretrained(args.model, use_safetensors=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.model)
    
    token = os.environ.get("HF_TOKEN")
    print("Loading facebook/winoground dataset from Hugging Face...")
    try:
        dataset = load_dataset("facebook/winoground", token=token, split="test")
    except Exception as e:
        print(f"Error loading Winoground dataset: {e}")
        print("Please make sure you are logged in to Hugging Face or set HF_TOKEN in your environment/.env file.")
        return
        
    num_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating {num_samples} samples...")
    
    # Initialize trackers for scores: Vanilla, Average, Product, Hybrid
    methods = ["vanilla", "mlqe_avg", "mlqe_prod", "mlqe_hybrid"]
    scores = {m: {"text": 0, "image": 0, "group": 0} for m in methods}
    results_log = []
    
    for idx in range(num_samples):
        sample = dataset[idx]
        img0 = sample["image_0"]
        img1 = sample["image_1"]
        cap0 = sample["caption_0"]
        cap1 = sample["caption_1"]
        
        # 1. Compute image features
        inputs_img0 = clip_processor(images=img0, return_tensors="pt").to(device)
        inputs_img1 = clip_processor(images=img1, return_tensors="pt").to(device)
        with torch.no_grad():
            emb_i0 = clip_model.get_image_features(**inputs_img0)
            emb_i0 = extract_tensor(emb_i0)
            emb_i0 = (emb_i0 / emb_i0.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()[0]
            
            emb_i1 = clip_model.get_image_features(**inputs_img1)
            emb_i1 = extract_tensor(emb_i1)
            emb_i1 = (emb_i1 / emb_i1.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()[0]
            
        # 2. Parse component features for M-LQE
        comps0 = parse_winoground_caption(cap0)
        comps1 = parse_winoground_caption(cap1)
        
        # Helper to encode text
        def get_text_emb(text):
            inputs_text = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                emb = clip_model.get_text_features(**inputs_text)
                emb = extract_tensor(emb)
                emb = (emb / emb.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()[0]
            return emb
            
        emb_c0 = get_text_emb(cap0)
        emb_c1 = get_text_emb(cap1)
        
        # Component embeddings
        emb_comps0 = [get_text_emb(f"a photo of a {comp}") for comp in comps0] if comps0 else [emb_c0]
        emb_comps1 = [get_text_emb(f"a photo of a {comp}") for comp in comps1] if comps1 else [emb_c1]
        
        # Calculate scores for each pair (Image X, Caption Y)
        pair_scores = {m: np.zeros((2, 2)) for m in methods}
        
        for i_idx, emb_i in enumerate([emb_i0, emb_i1]):
            for c_idx, (emb_c, emb_comps) in enumerate([(emb_c0, emb_comps0), (emb_c1, emb_comps1)]):
                # Vanilla similarity
                vanilla_sim = float(np.dot(emb_i, emb_c))
                pair_scores["vanilla"][i_idx, c_idx] = vanilla_sim
                
                # M-LQE similarities
                comp_sims = [float(np.dot(emb_i, ec)) for ec in emb_comps]
                avg_sim = float(np.mean(comp_sims))
                prod_sim = float(np.prod(comp_sims))
                
                pair_scores["mlqe_avg"][i_idx, c_idx] = avg_sim
                pair_scores["mlqe_prod"][i_idx, c_idx] = prod_sim
                
                # Hybrid similarity
                pair_scores["mlqe_hybrid"][i_idx, c_idx] = vanilla_sim + 0.5 * avg_sim
                
        # Evaluate metrics for each method
        sample_results = {"id": idx, "caption_0": cap0, "caption_1": cap1}
        for m in methods:
            m_scores = pair_scores[m]
            c0i0 = m_scores[0, 0]
            c1i0 = m_scores[0, 1]
            c0i1 = m_scores[1, 0]
            c1i1 = m_scores[1, 1]
            
            # Text Score: (c0i0 > c1i0) and (c1i1 > c0i1)
            text_correct = 1 if (c0i0 > c1i0) and (c1i1 > c0i1) else 0
            # Image Score: (c0i0 > c0i1) and (c1i1 > c1i0)
            image_correct = 1 if (c0i0 > c0i1) and (c1i1 > c1i0) else 0
            # Group Score: text and image correct
            group_correct = 1 if (text_correct and image_correct) else 0
            
            scores[m]["text"] += text_correct
            scores[m]["image"] += image_correct
            scores[m]["group"] += group_correct
            
            sample_results[f"{m}_text"] = text_correct
            sample_results[f"{m}_image"] = image_correct
            sample_results[f"{m}_group"] = group_correct
            
        results_log.append(sample_results)
        print(f"Sample {idx+1:03d} | Vanilla Group: {sample_results['vanilla_group']} | Avg Group: {sample_results['mlqe_avg_group']}")
        
    print("\n" + "=" * 50)
    print("WINOGROUND CLIP EVALUATION SUMMARY")
    print("=" * 50)
    summary_data = {}
    for m in methods:
        text_acc = scores[m]["text"] / num_samples
        image_acc = scores[m]["image"] / num_samples
        group_acc = scores[m]["group"] / num_samples
        print(f"Method: {m.upper()}")
        print(f"  Text Score  : {text_acc:.2%}")
        print(f"  Image Score : {image_acc:.2%}")
        print(f"  Group Score : {group_acc:.2%}")
        print("-" * 30)
        summary_data[m] = {
            "text_accuracy": text_acc,
            "image_accuracy": image_acc,
            "group_accuracy": group_acc
        }
        
    out_payload = {
        "dataset": "facebook/winoground",
        "num_samples": num_samples,
        "results": summary_data,
        "details": results_log
    }
    with open(args.output, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
