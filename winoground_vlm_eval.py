import os
import sys
# Force torchaudio to be treated as unavailable to prevent binary loading crashes
sys.modules['torchaudio'] = None

import re
import json
import argparse
import base64
import io
import random
import requests
from tqdm import tqdm
from PIL import Image
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

def query_vlm(img, prompt, provider, model, vlm_pipe, api_key, base_url):
    pred_text = ""
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
        res = vlm_pipe(messages, max_new_tokens=10)
        pred_text = res[0]["generated_text"]
        if isinstance(pred_text, list):
            try:
                pred_text = pred_text[-1]["content"]
            except (IndexError, KeyError, TypeError):
                pred_text = str(pred_text)
        else:
            pred_text = str(pred_text)
    else:
        # NIM provider
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
            "max_tokens": 10
        }
        
        invoke_url = f"{base_url.rstrip('/')}/chat/completions"
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        res_data = response.json()
        pred_text = res_data["choices"][0]["message"]["content"]
        
    return pred_text.strip()

def parse_choice(pred_text):
    clean_pred = re.sub(r"[^A-Za-z]", "", pred_text).upper()
    pred_label = ""
    if "A" in clean_pred and "B" not in clean_pred:
        pred_label = "A"
    elif "B" in clean_pred and "A" not in clean_pred:
        pred_label = "B"
    else:
        for char in clean_pred:
            if char in ["A", "B"]:
                pred_label = char
                break
    return pred_label

def create_composite_image(img0, img1):
    h = min(img0.height, img1.height)
    w0 = int(img0.width * (h / img0.height))
    w1 = int(img1.width * (h / img1.height))
    
    img0_resized = img0.resize((w0, h), Image.Resampling.LANCZOS)
    img1_resized = img1.resize((w1, h), Image.Resampling.LANCZOS)
    
    border_width = 10
    composite = Image.new("RGB", (w0 + border_width + w1, h), (0, 0, 0))
    composite.paste(img0_resized, (0, 0))
    composite.paste(img1_resized, (w0 + border_width, 0))
    return composite

def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="VLM Evaluation on Winoground")
    parser.add_argument("--provider", type=str, choices=["hf", "nim"], default="hf", help="Model API provider")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model key/name")
    parser.add_argument("--api-key", type=str, default="", help="NVIDIA NIM API key")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--output", type=str, default="winoground_vlm_results.json", help="Path to save result JSON")
    args = parser.parse_args()
    
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    if not args.api_key:
        args.api_key = os.environ.get("NVIDIA_API_KEY", "")
        
    if args.provider == "nim" and args.model == "Qwen/Qwen2-VL-2B-Instruct":
        args.model = "meta/llama-3.2-11b-vision-instruct"
        
    print(f"Evaluation Provider: {args.provider.upper()}")
    print(f"Model: {args.model}")
    
    vlm_pipe = None
    if args.provider == "hf":
        print("Initializing Hugging Face VLM pipeline...")
        from transformers import pipeline
        try:
            vlm_pipe = pipeline("image-text-to-text", model=args.model, device_map="auto")
        except Exception as e:
            print(f"Error loading Hugging Face pipeline: {e}")
            return
            
    token = os.environ.get("HF_TOKEN")
    print("Loading facebook/winoground dataset...")
    try:
        dataset = load_dataset("facebook/winoground", token=token, split="test")
    except Exception as e:
        print(f"Error loading Winoground: {e}")
        return
        
    num_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating {num_samples} samples...")
    
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    results_log = []
    
    # Random seed for choice shuffling
    random.seed(42)
    
    print("\n--- Starting Evaluation ---")
    for idx in range(num_samples):
        sample = dataset[idx]
        img0 = sample["image_0"]
        img1 = sample["image_1"]
        cap0 = sample["caption_0"]
        cap1 = sample["caption_1"]
        
        # --- A. Evaluate Text Score ---
        # 1. Image 0 matching: Choose between cap0 and cap1
        if random.choice([True, False]):
            opt_a, opt_b = cap0, cap1
            correct_label_i0 = "A"
        else:
            opt_a, opt_b = cap1, cap0
            correct_label_i0 = "B"
            
        prompt_i0 = (
            "Which of the following captions accurately describes the image? "
            "Choose Option A or Option B. Answer with ONLY the option letter (A or B) and nothing else.\n\n"
            f"Option A: {opt_a}\n"
            f"Option B: {opt_b}\n\n"
            "Your choice (A or B):"
        )
        
        # 2. Image 1 matching: Choose between cap0 and cap1
        if random.choice([True, False]):
            opt_a, opt_b = cap0, cap1
            correct_label_i1 = "B"
        else:
            opt_a, opt_b = cap1, cap0
            correct_label_i1 = "A"
            
        prompt_i1 = (
            "Which of the following captions accurately describes the image? "
            "Choose Option A or Option B. Answer with ONLY the option letter (A or B) and nothing else.\n\n"
            f"Option A: {opt_a}\n"
            f"Option B: {opt_b}\n\n"
            "Your choice (A or B):"
        )
        
        # --- B. Evaluate Image Score ---
        # We construct a composite image with img0 and img1
        # Shuffled sub-images
        img_a_is_img0 = random.choice([True, False])
        if img_a_is_img0:
            composite_img = create_composite_image(img0, img1)
            correct_label_c0 = "A"
            correct_label_c1 = "B"
        else:
            composite_img = create_composite_image(img1, img0)
            correct_label_c0 = "B"
            correct_label_c1 = "A"
            
        # 3. Caption 0 matching: Choose which sub-image matches cap0
        prompt_c0 = (
            "You are looking at a composite image containing two side-by-side sub-images. "
            "The left sub-image is Image A, and the right sub-image is Image B.\n"
            f"Which of these two sub-images matches the caption: '{cap0}'? "
            "Choose Option A or Option B. Answer with ONLY the option letter (A or B) and nothing else.\n\n"
            "Your choice (A or B):"
        )
        
        # 4. Caption 1 matching: Choose which sub-image matches cap1
        prompt_c1 = (
            "You are looking at a composite image containing two side-by-side sub-images. "
            "The left sub-image is Image A, and the right sub-image is Image B.\n"
            f"Which of these two sub-images matches the caption: '{cap1}'? "
            "Choose Option A or Option B. Answer with ONLY the option letter (A or B) and nothing else.\n\n"
            "Your choice (A or B):"
        )
        
        try:
            # Run VLM queries
            pred_i0 = query_vlm(img0, prompt_i0, args.provider, args.model, vlm_pipe, args.api_key, base_url)
            lbl_i0 = parse_choice(pred_i0)
            ok_i0 = 1 if lbl_i0 == correct_label_i0 else 0
            
            pred_i1 = query_vlm(img1, prompt_i1, args.provider, args.model, vlm_pipe, args.api_key, base_url)
            lbl_i1 = parse_choice(pred_i1)
            ok_i1 = 1 if lbl_i1 == correct_label_i1 else 0
            
            pred_c0 = query_vlm(composite_img, prompt_c0, args.provider, args.model, vlm_pipe, args.api_key, base_url)
            lbl_c0 = parse_choice(pred_c0)
            ok_c0 = 1 if lbl_c0 == correct_label_c0 else 0
            
            pred_c1 = query_vlm(composite_img, prompt_c1, args.provider, args.model, vlm_pipe, args.api_key, base_url)
            lbl_c1 = parse_choice(pred_c1)
            ok_c1 = 1 if lbl_c1 == correct_label_c1 else 0
            
            # Combine scores
            text_ok = 1 if (ok_i0 and ok_i1) else 0
            image_ok = 1 if (ok_c0 and ok_c1) else 0
            group_ok = 1 if (text_ok and image_ok) else 0
            
            text_correct_count += text_ok
            image_correct_count += image_ok
            group_correct_count += group_ok
            
            print(f"Sample {idx+1:03d} | Text Correct: {text_ok} | Image Correct: {image_ok} | Group Correct: {group_ok}")
            
            results_log.append({
                "id": idx,
                "caption_0": cap0,
                "caption_1": cap1,
                "text_correct": text_ok,
                "image_correct": image_ok,
                "group_correct": group_ok
            })
            
        except Exception as e:
            print(f"Error evaluating sample {idx+1}: {e}")
            continue
            
    text_acc = text_correct_count / num_samples if num_samples > 0 else 0
    image_acc = image_correct_count / num_samples if num_samples > 0 else 0
    group_acc = group_correct_count / num_samples if num_samples > 0 else 0
    
    print("\n" + "=" * 50)
    print("WINOGROUND VLM EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Samples Evaluated: {num_samples}")
    print(f"Text Score  : {text_acc:.2%}")
    print(f"Image Score : {image_acc:.2%}")
    print(f"Group Score : {group_acc:.2%}")
    print("=" * 50)
    
    out_payload = {
        "dataset": "facebook/winoground",
        "provider": args.provider,
        "model": args.model,
        "num_samples": num_samples,
        "results": {
            "text_accuracy": text_acc,
            "image_accuracy": image_acc,
            "group_accuracy": group_acc
        },
        "details": results_log
    }
    with open(args.output, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
