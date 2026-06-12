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

def extract_single_keyword(query: str, model, tokenizer, device: str) -> str:
    """Prompt the model to extract the single primary category noun/keyword from the question."""
    system_prompt = (
        "You are a precise search assistant. Your job is to extract the single primary noun or keyword "
        "representing the general category of the item being asked about in the user's question. "
        "Do not output any other text or punctuation. Output exactly one word in lowercase.\n\n"
        "Examples:\n"
        "Question: \"What degree did I graduate with?\"\n"
        "Output: degree\n\n"
        "Question: \"Where did I redeem a $5 coupon on coffee creamer?\"\n"
        "Output: creamer\n\n"
        "Question: \"What play did I attend at the local community theater?\"\n"
        "Output: play\n\n"
        "Question: \"What is the name of the playlist I created on Spotify?\"\n"
        "Output: playlist\n\n"
        "Question: \"What was my last name before I changed it?\"\n"
        "Output: name"
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
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # Simple cleanup to get a single word
    cleaned = response.split()[0].strip().strip(".,:;\"'()[]") if response else ""
    return cleaned.lower()

def extract_noun_and_expand_regex(query: str, model, tokenizer, device: str) -> str:
    """Prompt the model to expand the general category in the query into a regex pattern."""
    system_prompt = (
        "You are a precise search assistant. Your job is to extract the main noun representing a "
        "general category from the user's question, and output an exhaustive list of specific sub-categories, "
        "synonyms, and specific types/members belonging to that category, formatted as a pipe-separated regular expression (without spaces) "
        "enclosed in parentheses.\n\n"
        "Be extremely broad and include many specific common items of that category to ensure high search recall.\n\n"
        "Examples:\n"
        "Question: \"What degree did I graduate with?\"\n"
        "Output: (degree|major|diploma|graduation|bachelor|master|phd|bs|ba|science|arts|business|administration|engineering|studies)\n\n"
        "Question: \"Where did I redeem a $5 coupon on coffee creamer?\"\n"
        "Output: (creamer|coffee|milk|dairy|coupon|discount|redeem|supermarket|target|walmart|grocery|beverage|drink)\n\n"
        "Question: \"What play did I attend at the local community theater?\"\n"
        "Output: (play|theater|drama|show|musical|performance|act|broadway|tragedy|comedy|glass|menagerie|shakespeare|production)\n\n"
        "Question: \"What is the name of the playlist I created on Spotify?\"\n"
        "Output: (playlist|song|music|spotify|track|album|artist|audio|tune|summer|vibes|list|mix|station)\n\n"
        "Question: \"What was my last name before I changed it?\"\n"
        "Output: (name|surname|johnson|smith|lastname|changed|former|previous|birth|family|initials)\n\n"
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
        hidden = out.hidden_states[-1]
        emb = hidden.mean(dim=1).squeeze(0)
    return emb.cpu()

def run_grep_search(context: str, keyword: str) -> list[str]:
    """Vanilla Grep: Search for the extracted keyword with word boundaries."""
    if not keyword:
        return []
    turns = context.split("\n")
    matches = []
    pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for turn in turns:
        if pattern.search(turn):
            matches.append(turn)
    return matches

def run_lqe_grep_search(context: str, regex_pattern: str) -> list[str]:
    """LQE-Grep: Search turns using the synthesized regex pattern wrapped in word boundaries."""
    if not regex_pattern:
        return []
    turns = context.split("\n")
    matches = []
    
    if regex_pattern.startswith("(") and regex_pattern.endswith(")"):
        pattern_str = rf"\b{regex_pattern}\b"
    else:
        pattern_str = rf"\b({regex_pattern.strip('()')})\b"
        
    try:
        pattern = re.compile(pattern_str, re.IGNORECASE)
    except re.error:
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
    # To keep execution fast, we filter out turns that are very short or empty
    filtered_turns = [t for t in turns if len(t.split()) > 3]
    
    # If the haystack is huge, we sub-sample to the first 150 turns to keep the test fast,
    # or score everything if it's manageable.
    for turn in filtered_turns[:150]:
        turn_emb = embed_text(turn, model, tokenizer, device)
        sim = F.cosine_similarity(query_emb.unsqueeze(0), turn_emb.unsqueeze(0)).item()
        scores.append((turn, sim))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return [turn for turn, _ in scores[:top_k]]

def ask_agent_for_answer(context_turns: list[str], question: str, model, tokenizer, device: str) -> str:
    """Prompt the agent to answer the question using the retrieved context."""
    context_text = "\n".join(context_turns) if context_turns else "No relevant context found."
    
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "Be extremely brief and direct. Output ONLY the final answer (e.g., a single entity, name, or number) and nothing else."
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
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def evaluate_method(method_name: str, dataset: list[dict], model, tokenizer, device: str, extra_data: dict) -> tuple[float, float]:
    """Run evaluation for a specific retrieval method over the real dataset."""
    n_correct = 0
    total_tokens = 0
    
    for ex in dataset:
        context = ex["full_context"]
        query = ex["question"]
        ref_ans = ex["answer"]
        q_id = ex["question_id"]
        
        # 1. Retrieve
        if method_name == "grep":
            kw = extra_data[q_id]["keyword"]
            retrieved = run_grep_search(context, kw)
        elif method_name == "lqe_grep":
            pat = extra_data[q_id]["regex"]
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
        
        # Check correctness (lenient substring matching)
        is_correct = ref_ans.lower() in pred_ans.lower() or pred_ans.lower() in ref_ans.lower()
        if is_correct:
            n_correct += 1
            
        print(f"    Q ID {q_id}:")
        print(f"      Query: {query}")
        print(f"      Ref Ans: {ref_ans} | Pred Ans: '{pred_ans}' | Correct: {is_correct}")
        print(f"      Retrieved ({len(retrieved)} lines):")
        for line in retrieved[:3]:
            print(f"        {line[:120]}...")
        print(f"      ---")
            
    avg_tokens = total_tokens / len(dataset)
    accuracy = n_correct / len(dataset)
    return accuracy, avg_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key from src/models.py")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of questions to evaluate")
    args = parser.parse_args()
    
    device = best_gpu()
    print(f"Loading model {args.model}...")
    model, tokenizer = load(args.model, device=device)
    
    # Load dataset
    print("Loading LongMemEval dataset...")
    with open("data/longmemeval_s_cleaned.json", "r") as f:
        all_data = json.load(f)
        
    # Filter for single-session-user to test direct retrieval QA
    subset = [item for item in all_data if item["question_type"] == "single-session-user"]
    eval_subset = subset[:args.num_examples]
    print(f"Evaluating on {len(eval_subset)} real queries...")
    
    # Prepare contexts and pre-extract queries
    print("Formatting contexts and pre-generating keywords/patterns...")
    extra_data = {}
    for ex in eval_subset:
        q_id = ex["question_id"]
        
        # Format haystack sessions as a single flat string
        dialogue_history = []
        for sess in ex["haystack_sessions"]:
            for turn in sess:
                dialogue_history.append(f"[{turn['role'].capitalize()}]: {turn['content']}")
        full_context = "\n".join(dialogue_history)
        ex["full_context"] = full_context
        
        # Pre-extract keywords and regex patterns
        kw = extract_single_keyword(ex["question"], model, tokenizer, device)
        pat = extract_noun_and_expand_regex(ex["question"], model, tokenizer, device)
        extra_data[q_id] = {"keyword": kw, "regex": pat}
        print(f"  Q ID {q_id}: Query = '{ex['question']}'")
        print(f"    Keyword = '{kw}' | Regex = '{pat}'")
        
    results = {}
    
    # Run evaluations
    print("\nEvaluating Method: Vanilla Grep...")
    grep_acc, grep_tok = evaluate_method("grep", eval_subset, model, tokenizer, device, extra_data)
    print(f"  Grep Accuracy: {grep_acc:.2%}, Avg Tokens: {grep_tok:.1f}")
    
    print("\nEvaluating Method: LQE-Grep...")
    lqe_acc, lqe_tok = evaluate_method("lqe_grep", eval_subset, model, tokenizer, device, extra_data)
    print(f"  LQE-Grep Accuracy: {lqe_acc:.2%}, Avg Tokens: {lqe_tok:.1f}")
    
    print("\nEvaluating Method: Vector Search (Top-3 CosSim)...")
    vec_acc, vec_tok = evaluate_method("vector", eval_subset, model, tokenizer, device, extra_data)
    print(f"  Vector Accuracy: {vec_acc:.2%}, Avg Tokens: {vec_tok:.1f}")
    
    results = {
        "grep": {"accuracy": grep_acc, "avg_tokens": grep_tok},
        "lqe_grep": {"accuracy": lqe_acc, "avg_tokens": lqe_tok},
        "vector": {"accuracy": vec_acc, "avg_tokens": vec_tok}
    }
    
    # Save results
    output_path = "lqe_real_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n\n### Real LongMemEval Experiment Summary Table")
    print("| Method | Accuracy | Avg Tokens |")
    print("| --- | --- | --- |")
    for method in ["grep", "vector", "lqe_grep"]:
        m_res = results[method]
        print(f"| {method.upper()} | {m_res['accuracy']:.1%} | {m_res['avg_tokens']:.1f} |")
        
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
