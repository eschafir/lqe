import os
import sys
# Force torchaudio to be treated as unavailable to prevent binary loading crashes
sys.modules['torchaudio'] = None

import re
import json
import argparse
import base64
import io
import math
import requests
import numpy as np
import torch
from PIL import Image, ImageDraw
from datasets import load_dataset

CACHE_FILE = "vlm_captions_cache.json"

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

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save caption cache: {e}")

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

def create_toy_image(shape, color, bg_color="white", size=(224, 224)):
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    w, h = size
    pad = 40
    if shape == "circle":
        draw.ellipse([pad, pad, w - pad, h - pad], fill=color, outline="black", width=2)
    elif shape == "square":
        draw.rectangle([pad, pad, w - pad, h - pad], fill=color, outline="black", width=2)
    elif shape == "triangle":
        points = [(w // 2, pad), (pad, h - pad), (w - pad, h - pad)]
        draw.polygon(points, fill=color, outline="black", width=2)
    elif shape == "star":
        cx, cy = w // 2, h // 2
        r_outer = 80
        r_inner = 35
        points = []
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = i * math.pi / 5 - math.pi / 2
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color, outline="black", width=2)
    else:
        draw.rectangle([pad, pad + 20, w - pad, h - pad - 20], fill=color, outline="black", width=2)
    return img

def setup_toy_dataset():
    toy_configs = [
        {"shape": "circle", "color": "red", "desc": "a red circle"},
        {"shape": "circle", "color": "blue", "desc": "a blue circle"},
        {"shape": "circle", "color": "green", "desc": "a green circle"},
        {"shape": "triangle", "color": "yellow", "desc": "a yellow triangle"},
        {"shape": "triangle", "color": "purple", "desc": "a purple triangle"},
        {"shape": "triangle", "color": "red", "desc": "a red triangle"},
        {"shape": "square", "color": "blue", "desc": "a blue square"},
        {"shape": "square", "color": "green", "desc": "a green square"},
        {"shape": "square", "color": "yellow", "desc": "a yellow square"},
        {"shape": "star", "color": "purple", "desc": "a purple star"}
    ]
    dataset = []
    for idx, cfg in enumerate(toy_configs):
        img = create_toy_image(cfg["shape"], cfg["color"])
        dataset.append({
            "id": idx,
            "image": img,
            "description": cfg["desc"]
        })
    return dataset

# LQE expansion logic
def local_lqe_expand(query: str) -> str:
    synonyms = {
        "red": ["red", "crimson", "scarlet", "ruby"],
        "blue": ["blue", "azure", "navy", "cyan", "sapphire"],
        "green": ["green", "emerald", "lime", "olive"],
        "yellow": ["yellow", "gold", "amber", "lemon"],
        "purple": ["purple", "violet", "magenta", "plum"],
        "circle": ["circle", "round", "ring", "disk"],
        "square": ["square", "block", "box", "rectangle"],
        "triangle": ["triangle", "pyramid", "wedge"],
        "star": ["star", "asterisk", "pentagram"],
        "door": ["door", "gate", "entry", "entrance"],
        "man": ["man", "person", "guy", "male", "gentleman"],
        "open": ["open", "opened", "unlocked"],
        "crouched": ["crouched", "crouching", "bent", "stooped"],
        "banana": ["banana", "fruit"],
        "plate": ["plate", "dish", "saucer"],
        "table": ["table", "desk", "bench"],
        "street": ["street", "road", "lane", "path"],
        "wall": ["wall", "partition", "barrier"]
    }
    
    words = re.findall(r"\b\w+\b", query.lower())
    regex_parts = []
    
    for word in words:
        if word in ["a", "an", "the", "and", "in", "on", "next", "to"]:
            continue
        if word in synonyms:
            group = "|".join(synonyms[word])
            regex_parts.append(f"\\b({group})\\b")
        else:
            regex_parts.append(f"\\b{word}\\b")
            
    return ".*".join(regex_parts)

def parse_caption(caption: str) -> list[str]:
    delimiters = r"\b(and|in|on|under|next to|near|behind|in front of|at|with|over|above|below|through|inside|into)\b"
    parts = re.split(delimiters, caption, flags=re.IGNORECASE)
    components = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\b(the|a|an|some|any)\b\s*", "", p, flags=re.IGNORECASE)
        p = re.sub(r"[^\w\s]", "", p).strip()
        if p and p not in ["and", "in", "on", "under", "next to", "near", "behind", "in front of", "at", "with", "over", "above", "below", "through", "inside", "into"]:
            components.append(p)
    return components

def get_vlm_caption(img, idx, dataset_name, provider, model, vlm_pipe, api_key, base_url, cache):
    cache_key = f"{dataset_name}_{idx}"
    if cache_key in cache:
        return cache[cache_key]
        
    print(f"  [VLM] Captioning image {idx} on-the-fly...")
    prompt = "Describe this image in detail. Be precise about the colors, shapes, objects, and attributes you see."
    
    caption = ""
    try:
        if provider == "hf":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            res = vlm_pipe(messages, max_new_tokens=40)
            caption = res[0]["generated_text"]
            if isinstance(caption, list):
                caption = caption[-1]["content"]
            else:
                caption = str(caption)
        else:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            headers = {
                "Content-Type": "application/json"
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 40
            }
            invoke_url = f"{base_url.rstrip('/')}/chat/completions"
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            res_data = response.json()
            caption = res_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Warning: VLM captioning failed: {e}. Falling back to default description.")
        caption = f"image {idx}"
        
    caption = caption.strip()
    cache[cache_key] = caption
    save_cache(cache)
    return caption

def main():
    load_env_file()
    cache = load_cache()
    
    print("=" * 60)
    print("        INTERACTIVE MULTIMODAL RETRIEVAL PIPELINE")
    print("=" * 60)
    
    # 1. Dataset Selection
    print("\nSelect a Dataset:")
    print("  1. Toy Shapes Dataset (10 images - Instant)")
    print("  2. ARO Visual Attribution (20 images - Loads from HF)")
    print("  3. Winoground (20 images - Loads from HF)")
    dataset_choice = input("Enter choice (1-3) [default: 1]: ").strip()
    if not dataset_choice:
        dataset_choice = "1"
        
    # 2. Method Selection
    print("\nSelect Retrieval Method:")
    print("  1. Vanilla CLIP (Global Similarity)")
    print("  2. M-LQE (Average Similarity)")
    print("  3. M-LQE (Product Similarity)")
    print("  4. M-LQE (Hybrid Similarity)")
    print("  5. VLM (Caption-RAG / Regex Match)")
    method_choice = input("Enter choice (1-5) [default: 1]: ").strip()
    if not method_choice:
        method_choice = "1"
        
    # 3. Enter Search Caption
    default_queries = {
        "1": "a purple star",
        "2": "the open door and the crouched man",
        "3": "some person standing in a forest"
    }
    default_q = default_queries.get(dataset_choice, "a photo of an object")
    query_text = input(f"\nEnter your query caption [default: '{default_q}']: ").strip()
    if not query_text:
        query_text = default_q
        
    # Load settings
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = []
    dataset_name = ""
    if dataset_choice == "1":
        dataset = setup_toy_dataset()
        dataset_name = "toy"
    elif dataset_choice == "2":
        dataset_name = "aro"
        print("Loading ARO Visual Attribution from Hugging Face...")
        ds = load_dataset("gowitheflow/ARO-Visual-Attribution", split="test", streaming=True)
        for idx, sample in enumerate(ds):
            if idx >= 20:
                break
            dataset.append({
                "id": idx,
                "image": sample["image"],
                "description": sample["true_caption"]
            })
    elif dataset_choice == "3":
        dataset_name = "winoground"
        token = os.environ.get("HF_TOKEN")
        print("Loading Winoground from Hugging Face...")
        ds = load_dataset("facebook/winoground", token=token, split="test")
        for idx in range(min(20, len(ds))):
            sample = ds[idx]
            dataset.append({
                "id": idx * 2,
                "image": sample["image_0"],
                "description": sample["caption_0"]
            })
            dataset.append({
                "id": idx * 2 + 1,
                "image": sample["image_1"],
                "description": sample["caption_1"]
            })
            
    print(f"Dataset loaded. Total pool size: {len(dataset)} images.")
    
    # Run retrieval
    scores = []
    
    if method_choice == "5":
        # VLM Caption-RAG
        print("\nUsing VLM (Caption-RAG) Method...")
        # Check if we need VLM pipeline (if HF) or NIM
        provider = "nim" if api_key else "hf"
        model = "meta/llama-3.2-11b-vision-instruct" if provider == "nim" else "Qwen/Qwen2-VL-2B-Instruct"
        vlm_pipe = None
        
        if provider == "hf":
            print(f"No NVIDIA API key found. Initializing local Hugging Face VLM pipeline ({model})...")
            from transformers import pipeline
            vlm_pipe = pipeline("image-text-to-text", model=model, device_map="auto")
            
        # 1. Expand query to regex pattern
        regex_pattern = local_lqe_expand(query_text)
        print(f"  Synthesizing Lexical Query Expansion Regex: '{regex_pattern}'")
        pattern = re.compile(regex_pattern, re.IGNORECASE)
        
        # 2. Get/Generate captions and run regex matching
        print("  Scanning database captions...")
        for item in dataset:
            # Special case for toy: use pre-defined captions to avoid API costs
            if dataset_name == "toy":
                caption = item["description"]
            else:
                caption = get_vlm_caption(
                    item["image"], item["id"], dataset_name, 
                    provider, model, vlm_pipe, api_key, base_url, cache
                )
                
            # Perform regex search
            match = pattern.search(caption)
            score = 1.0 if match else 0.0
            
            # If matches, we can also compute a secondary CLIP score or text overlap
            # for sorting, otherwise we sort by boolean match
            scores.append((item, score, caption))
            
    else:
        # CLIP based methods
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nInitializing CLIP on {device.upper()}...")
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 1. Compute query representations
        def get_text_emb(text):
            inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                emb = clip_model.get_text_features(**inputs)
                emb = extract_tensor(emb)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.cpu().numpy()[0]
            
        print("Encoding query representations...")
        query_emb = get_text_emb(query_text)
        
        # For M-LQE variants, encode components
        comps = parse_caption(query_text)
        comp_embs = [get_text_emb(f"a photo of a {c}") for c in comps] if comps else [query_emb]
        print(f"  Query decomposed into components: {comps}")
        
        # 2. Compute image representations and calculate similarities
        print("Encoding database images and ranking...")
        for item in dataset:
            inputs_img = clip_processor(images=item["image"], return_tensors="pt").to(device)
            with torch.no_grad():
                emb_i = clip_model.get_image_features(**inputs_img)
                emb_i = extract_tensor(emb_i)
                emb_i = emb_i / emb_i.norm(p=2, dim=-1, keepdim=True)
                emb_i = emb_i.cpu().numpy()[0]
                
            # Compute matching score based on selected method
            if method_choice == "1":
                # Vanilla CLIP
                score = float(np.dot(emb_i, query_emb))
            elif method_choice == "2":
                # M-LQE Average
                score = float(np.mean([np.dot(emb_i, ce) for ce in comp_embs]))
            elif method_choice == "3":
                # M-LQE Product
                score = float(np.prod([np.dot(emb_i, ce) for ce in comp_embs]))
            else:
                # M-LQE Hybrid
                vanilla_score = float(np.dot(emb_i, query_emb))
                avg_score = float(np.mean([np.dot(emb_i, ce) for ce in comp_embs]))
                score = vanilla_score + 0.5 * avg_score
                
            scores.append((item, score, item["description"]))
            
    # Sort results
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Display Results
    print(f"\n=======================================================")
    print(f"               RETRIEVAL RANKING RESULTS")
    print(f"=======================================================")
    print(f"Query: '{query_text}'")
    print("-" * 55)
    
    for rank, (item, score, description) in enumerate(scores[:3]):
        print(f"Rank {rank+1}: Sample ID {item['id']} | Match Score = {score:.4f}")
        print(f"        Ground Truth Description: '{item['description']}'")
        if method_choice == "5":
            print(f"        VLM Caption Scan        : '{description}'")
        print("-" * 55)
        
    # Save top-1 image
    top_item = scores[0][0]
    top_score = scores[0][1]
    top_image = top_item["image"]
    output_path = "retrieved_output.png"
    top_image.save(output_path)
    
    if method_choice == "5" and top_score == 0.0:
        print("\nWARNING: None of the VLM captions in the database matched the query regex pattern!")
        print(f"No match found. Saving the first image (Sample ID {top_item['id']}) as a fallback to: {output_path}")
    else:
        print(f"\nSUCCESS! Top retrieved image (ID {top_item['id']}) saved to: {output_path}")

if __name__ == "__main__":
    main()
