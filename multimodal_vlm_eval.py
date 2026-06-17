import os
import json
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation on ARO Visual Attribution")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Pretrained VLM model key")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of test samples to evaluate")
    parser.add_argument("--output", type=str, default="vlm_aro_results.json", help="Path to save result JSON")
    args = parser.parse_args()
    
    print(f"Initializing VLM pipeline for {args.model}...")
    try:
        # Load the VLM using the high-level image-text-to-text pipeline
        vlm_pipe = pipeline("image-text-to-text", model=args.model, device_map="auto")
        print("VLM pipeline successfully initialized.")
    except Exception as e:
        print(f"Error loading VLM pipeline: {e}")
        print("Make sure you have transformers, accelerate, and flash-attn (optional) installed.")
        return
        
    # Load ARO Visual Attribution dataset from Hugging Face
    print("Loading ARO Visual Attribution dataset from HF...")
    dataset = load_dataset("gowitheflow/ARO-Visual-Attribution", split="test", streaming=True)
    
    correct_count = 0
    total_count = 0
    results_log = []
    
    # Set seed for reproducibility of option shuffling
    random.seed(42)
    
    print("\n--- Starting VLM Evaluation ---")
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
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            res = vlm_pipe(messages, max_new_tokens=10)
            pred_text = res[0]["generated_text"].strip()
            
            # Simple parsing of response: check if A or B is in the output
            # Clean up punctuation
            clean_pred = re.sub(r"[^A-Za-z]", "", pred_text).upper()
            
            # Extract first character or search for A/B
            pred_label = ""
            if "A" in clean_pred and "B" not in clean_pred:
                pred_label = "A"
            elif "B" in clean_pred and "A" not in clean_pred:
                pred_label = "B"
            else:
                # Fallback: check first character
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
    print("VLM Evaluation Summary on ARO Attribution")
    print("=" * 50)
    print(f"Total Samples Evaluated: {total_count}")
    print(f"VLM Accuracy:             {vlm_accuracy:.2%}")
    print("=" * 50)
    
    # Save results to JSON
    summary = {
        "dataset": "gowitheflow/ARO-Visual-Attribution",
        "model": args.model,
        "num_samples": total_count,
        "vlm_accuracy": vlm_accuracy,
        "details": results_log
    }
    
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    import re
    main()
