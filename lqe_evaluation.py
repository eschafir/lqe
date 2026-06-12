"""
Lexical Query Expansion (LQE) Evaluation Script.

This script benchmarks three agent search methods on the synthetic memory dialogue dataset:
  1. Vanilla Grep: Exact string matching using the general category noun.
  2. LQE-Grep: Synthesizing a high-recall regex pattern using a local LLM, then running grep.
  3. Vector Search: Computing embeddings of each turn using Qwen's internal hidden states.

The script sweeps across noise levels (number of distractors) to analyze:
  - QA Accuracy (correctness of final answer)
  - Context Footprint (number of tokens sent to the agent's context window)
"""
import argparse
import json
import os
import re
import sys
import time
import torch
import torch.nn.functional as F

# Add project root to sys.path to find src.models
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load, best_gpu
from synthetic_memory import generate_benchmark_dataset, CATEGORIES


def extract_noun_and_expand_regex(query: str, model, tokenizer, device: str) -> str:
    """Prompt the model to expand the general category in the query into a regex pattern."""
    system_prompt = (
        "You are a precise search assistant. Your job is to extract the main noun representing a "
        "general category from the user's question, and output an exhaustive list of specific sub-categories, "
        "synonyms, and specific types/members belonging to that category, formatted as a pipe-separated regular expression (without spaces) "
        "enclosed in parentheses.\n\n"
        "Be extremely broad and include many specific common items of that category to ensure high search recall.\n\n"
        "Examples:\n"
        "Question: \"What color vehicle did the user buy?\"\n"
        "Output: (sedan|coupe|motorcycle|suv|truck|hatchback|convertible|automobile|vehicle|car|bike|scooter|van|wagon)\n\n"
        "Question: \"What occupation did the user get?\"\n"
        "Output: (job|career|profession|doctor|engineer|plumber|barista|teacher|artist|nurse|programmer|physician|carpenter|mechanic|accountant|occupation|work)\n\n"
        "Question: \"What type of residence did the user move into?\"\n"
        "Output: (residence|dwelling|house|apartment|condo|townhouse|loft|cottage|penthouse|bungalow|cabin|studio|home|mansion|villa)\n\n"
        "Question: \"What type of pet did the user adopt?\"\n"
        "Output: (pet|animal|canine|feline|dog|cat|rabbit|hamster|parakeet|goldfish|puppy|kitten|bird|fish|guinea|spaniel|siamese)\n\n"
        "Question: \"What type of refreshment did the user order?\"\n"
        "Output: (refreshment|beverage|drink|liquid|water|coffee|tea|juice|soda|lemonade|smoothie|espresso|latte|macchiato|kombucha|beer|wine|milkshake)\n\n"
        "Question: \"What piece of apparel did the user buy?\"\n"
        "Output: (apparel|clothing|garment|clothes|wear|suit|shirt|pants|jeans|dress|coat|jacket|blazer|cardigan|sweater|trousers|sneakers|shoes|socks|trenchcoat|pullover)\n\n"
        "Respond ONLY with the final pipe-separated regex pattern enclosed in parentheses, like (type1|type2|type3). "
        "Do not output any other text."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: \"{query}\"\nOutput:"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Extract the pattern enclosed in parentheses
    match = re.search(r"(\([a-zA-Z0-9|_-]+\))", response)
    if match:
        return match.group(1)
    
    # Fallback cleanup
    cleaned = response.replace(" ", "").replace("\n", "")
    if not cleaned.startswith("("):
        cleaned = f"({cleaned})"
    return cleaned


def embed_text(text: str, model, tokenizer, device: str) -> torch.Tensor:
    """Generate mean-pooled embedding from Qwen's last hidden state layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.model(**inputs, output_hidden_states=True)
        # Mean pooling of the last hidden state
        hidden = out.hidden_states[-1] # (1, seq_len, hidden_size)
        emb = hidden.mean(dim=1).squeeze(0) # (hidden_size,)
    return emb.cpu()


def run_grep_search(context: str, category: str) -> list[str]:
    """Vanilla Grep: Search for the exact category keyword (will fail due to vocab mismatch)."""
    turns = context.split("\n")
    matches = []
    # Search for category keyword case-insensitively
    pattern = re.compile(rf"\b{category}\b", re.IGNORECASE)
    for turn in turns:
        if pattern.search(turn):
            matches.append(turn)
    return matches


def run_lqe_grep_search(context: str, regex_pattern: str) -> list[str]:
    """LQE-Grep: Search turns using the synthesized regex pattern."""
    turns = context.split("\n")
    matches = []
    
    # Wrap in word boundaries to prevent substring matches (e.g. 'car' matching 'cardigan')
    if regex_pattern.startswith("(") and regex_pattern.endswith(")"):
        pattern_str = rf"\b{regex_pattern}\b"
    else:
        pattern_str = rf"\b({regex_pattern.strip('()')})\b"
        
    try:
        pattern = re.compile(pattern_str, re.IGNORECASE)
    except re.error:
        # Fallback if regex generation is malformed
        return []
    
    for turn in turns:
        if pattern.search(turn):
            matches.append(turn)
    return matches


def run_vector_search(context: str, query: str, model, tokenizer, device: str, top_k: int = 3) -> list[str]:
    """Vector Search: Score turns by cosine similarity of pooled hidden states."""
    turns = context.split("\n")
    if not turns:
        return []
        
    query_emb = embed_text(query, model, tokenizer, device)
    
    scores = []
    for turn in turns:
        turn_emb = embed_text(turn, model, tokenizer, device)
        sim = F.cosine_similarity(query_emb.unsqueeze(0), turn_emb.unsqueeze(0)).item()
        scores.append((turn, sim))
        
    # Sort by similarity desc
    scores.sort(key=lambda x: x[1], reverse=True)
    return [turn for turn, _ in scores[:top_k]]


def ask_agent_for_answer(context_turns: list[str], question: str, model, tokenizer, device: str) -> str:
    """Prompt the agent to answer the question using the retrieved context."""
    context_text = "\n".join(context_turns) if context_turns else "No relevant context found."
    
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "Be brief and direct. Output ONLY the final answer (e.g., a single color or word) and nothing else."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate_method(method_name: str, dataset: list[dict], model, tokenizer, device: str, expanded_patterns: dict) -> tuple[float, float]:
    """Run evaluation for a specific retrieval method over the dataset."""
    n_correct = 0
    total_tokens = 0
    
    for ex in dataset:
        context = ex["context"]
        query = ex["query"]
        category = ex["category"]
        ref_ans = ex["reference_answer"]
        
        # 1. Retrieve
        if method_name == "grep":
            retrieved = run_grep_search(context, category)
        elif method_name == "lqe_grep":
            pat = expanded_patterns[ex["id"]]
            retrieved = run_lqe_grep_search(context, pat)
        elif method_name == "vector":
            retrieved = run_vector_search(context, query, model, tokenizer, device, top_k=3)
        else:
            retrieved = []
            
        # 2. Measure retrieved tokens
        retrieved_text = "\n".join(retrieved)
        tokens = len(tokenizer.tokenize(retrieved_text))
        total_tokens += tokens
        
        # 3. QA
        pred_ans = ask_agent_for_answer(retrieved, query, model, tokenizer, device)
        
        # Check correctness (lenient case-insensitive matching)
        is_correct = ref_ans.lower() in pred_ans.lower()
        if is_correct:
            n_correct += 1
            
        print(f"    Ex ID {ex['id']} ({category}):")
        print(f"      Query: {query}")
        print(f"      Ref Ans: {ref_ans} | Pred Ans: '{pred_ans}' | Correct: {is_correct}")
        print(f"      Retrieved ({len(retrieved)} lines):")
        for line in retrieved:
            print(f"        {line}")
        print(f"      ---")
            
    avg_tokens = total_tokens / len(dataset)
    accuracy = n_correct / len(dataset)
    return accuracy, avg_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key from src/models.py")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples per category (6 categories total)")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "lqe_results.json"), help="Path to save result JSON")
    args = parser.parse_args()
    
    device = best_gpu()
    print(f"Loading model {args.model}...")
    model, tokenizer = load(args.model, device=device)
    
    # We will test three noise levels: 10, 20, and 35 distractor sentences
    noise_levels = [10, 20, 35]
    results = {}
    
    # 1. Pre-generate the dataset for the largest noise level so we can expand patterns once
    # (Since queries are identical across noise levels, LQE patterns are generated once per query ID)
    print("\nGenerating base queries and synthesizing LQE regex patterns...")
    temp_dataset = generate_benchmark_dataset(num_examples_per_cat=args.num_examples, num_distractors=10, base_seed=42)
    
    expanded_patterns = {}
    for ex in temp_dataset:
        pat = extract_noun_and_expand_regex(ex["query"], model, tokenizer, device)
        expanded_patterns[ex["id"]] = pat
        print(f"  Query ID {ex['id']} ({ex['category']}): Query = '{ex['query']}' -> Regex = {pat}")
        
    # Sweep over noise levels
    for noise in noise_levels:
        print(f"\n==========================================")
        print(f"Running evaluation with {noise} distractor turns...")
        print(f"==========================================")
        
        dataset = generate_benchmark_dataset(num_examples_per_cat=args.num_examples, num_distractors=noise, base_seed=42)
        
        # Eval Grep
        print("Evaluating Method: Vanilla Grep...")
        grep_acc, grep_tok = evaluate_method("grep", dataset, model, tokenizer, device, expanded_patterns)
        print(f"  Grep Accuracy: {grep_acc:.2%}, Avg Tokens: {grep_tok:.1f}")
        
        # Eval LQE-Grep
        print("Evaluating Method: LQE-Grep...")
        lqe_acc, lqe_tok = evaluate_method("lqe_grep", dataset, model, tokenizer, device, expanded_patterns)
        print(f"  LQE-Grep Accuracy: {lqe_acc:.2%}, Avg Tokens: {lqe_tok:.1f}")
        
        # Eval Vector
        print("Evaluating Method: Vector Search (Hidden States CosSim)...")
        vec_acc, vec_tok = evaluate_method("vector", dataset, model, tokenizer, device, expanded_patterns)
        print(f"  Vector Accuracy: {vec_acc:.2%}, Avg Tokens: {vec_tok:.1f}")
        
        results[str(noise)] = {
            "grep": {"accuracy": grep_acc, "avg_tokens": grep_tok},
            "lqe_grep": {"accuracy": lqe_acc, "avg_tokens": lqe_tok},
            "vector": {"accuracy": vec_acc, "avg_tokens": vec_tok}
        }
        
    # Write JSON results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    # Print a beautiful Markdown Summary Table
    print("\n\n### Experiment Summary Table")
    print("| Noise (Distractors) | Method | Accuracy | Avg Tokens |")
    print("| --- | --- | --- | --- |")
    for noise in noise_levels:
        for method in ["grep", "lqe_grep", "vector"]:
            m_res = results[str(noise)][method]
            print(f"| {noise} | {method.upper()} | {m_res['accuracy']:.1%} | {m_res['avg_tokens']:.1f} |")
            
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
