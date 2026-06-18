import os
import sys
# Force torchaudio to be treated as unavailable to prevent binary loading crashes
sys.modules['torchaudio'] = None

import re
import json
import random
import argparse
import base64
import io
import requests
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

def load_env_file(dotenv_path=".env"):
    """Reads a local .env file and sets key-value pairs in os.environ without requiring external packages."""
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

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_env_file(os.path.join(project_root, ".env"))
    
    parser = argparse.ArgumentParser(description="VLM/NIM Evaluation on ARO Visual Attribution")
    parser.add_argument("--provider", type=str, choices=["hf", "nim"], default="hf", help="Model API provider")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model key/name")
    parser.add_argument("--api-key", type=str, default="", help="NVIDIA NIM API key")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--output", type=str, default=os.path.join(project_root, "results", "vlm_aro_results.json"), help="Path to save result JSON")
    args = parser.parse_args()
    
    # Hardcode base URL to NVIDIA Cloud API Catalog with environment override fallback
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    # Fallback to environment variable if argument is not provided
    if not args.api_key:
        args.api_key = os.environ.get("NVIDIA_API_KEY", "")
        
    # Adjust default model for NIM if unchanged
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
            print("Hugging Face pipeline successfully initialized.")
        except Exception as e:
            print(f"Error loading Hugging Face pipeline: {e}")
            return
    else:
        print(f"Connecting to NVIDIA NIM endpoint at: {base_url}")
        
    # Load ARO Visual Attribution dataset from Hugging Face
    print("Loading ARO Visual Attribution dataset from HF...")
    dataset = load_dataset("gowitheflow/ARO-Visual-Attribution", split="test", streaming=True)
    
    correct_count = 0
    total_count = 0
    results_log = []
    
    # Set seed for reproducibility of option shuffling
    random.seed(42)
    
    print("\n--- Starting Evaluation ---")
    for idx, sample in enumerate(dataset):
        if total_count >= args.num_samples:
            break
            
        img = sample["image"]
        true_cap = sample["true_caption"]
        false_cap = sample["false_caption"]
        
        # Shuffle options to avoid position bias
        if random.choice([True, False]):
            option_a = true_cap
            option_b = false_cap
            correct_label = "A"
        else:
            option_a = false_cap
            option_b = true_cap
            correct_label = "B"
            
        prompt = (
            "Which of the following captions accurately describes the image? "
            "Choose Option A or Option B. Answer with ONLY the option letter (A or B) and nothing else.\n\n"
            f"Option A: {option_a}\n"
            f"Option B: {option_b}\n\n"
            "Your choice (A or B):"
        )
        
        pred_text = ""
        try:
            if args.provider == "hf":
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
                # NIM provider: encode image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                headers = {
                    "Content-Type": "application/json"
                }
                if args.api_key:
                    headers["Authorization"] = f"Bearer {args.api_key}"
                    
                payload = {
                    "model": args.model,
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
                
            pred_text = pred_text.strip()
            
            # Simple parsing of response: check if A or B is in the output
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
            
            is_correct = 1 if pred_label == correct_label else 0
            if is_correct:
                correct_count += 1
                
            total_count += 1
            print(f"Sample {total_count:02d} | Pred: '{pred_label}' (Model Output: '{pred_text}') | Correct: '{correct_label}' | Match: {is_correct}")
            
            results_log.append({
                "id": total_count,
                "true_caption": true_cap,
                "false_caption": false_cap,
                "option_a": option_a,
                "option_b": option_b,
                "correct_label": correct_label,
                "pred_label": pred_label,
                "raw_output": pred_text,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"Error evaluating sample {total_count+1}: {e}")
            continue
            
    vlm_accuracy = (correct_count / total_count) if total_count > 0 else 0.0
    
    print("\n" + "=" * 50)
    print("VLM/NIM Evaluation Summary on ARO Attribution")
    print("=" * 50)
    print(f"Total Samples Evaluated: {total_count}")
    print(f"Accuracy:                 {vlm_accuracy:.2%}")
    print("=" * 50)
    
    # Save results to JSON
    summary = {
        "dataset": "gowitheflow/ARO-Visual-Attribution",
        "provider": args.provider,
        "model": args.model,
        "num_samples": total_count,
        "accuracy": vlm_accuracy,
        "details": results_log
    }
    
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
