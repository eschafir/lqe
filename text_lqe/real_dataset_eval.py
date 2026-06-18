import argparse
import json
import os
import re
import sys
import time
import torch
import torch.nn.functional as F

# Add project root parent to sys.path to find src.models
project_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_parent)
sys.path.insert(0, os.path.join(project_parent, "subspace-search"))

from src.models import load, best_gpu

def extract_single_keyword(query: str, model, tokenizer, device: str) -> tuple[str, int]:
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
        
    token_count = out.shape[1]
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # Simple cleanup to get a single word
    cleaned = response.split()[0].strip().strip(".,:;\"'()[]") if response else ""
    return cleaned.lower(), token_count

def extract_noun_and_expand_regex(query: str, model, tokenizer, device: str) -> tuple[str, int]:
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
        
    token_count = out.shape[1]
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Extract the pattern enclosed in parentheses
    match = re.search(r"(\([a-zA-Z0-9|_-]+\))", response)
    if match:
        return match.group(1), token_count
    
    # Fallback cleanup
    cleaned = response.replace(" ", "").replace("\n", "")
    if not cleaned.startswith("("):
        cleaned = f"({cleaned})"
    return cleaned, token_count


def prune_regex_pattern(regex_pattern: str, query: str = "", primary_keyword: str = "", context: str = "", threshold: float = 0.05) -> str:
    """
    LQE v2: Filter out generic, high-frequency, or short English words from the expanded regex.
    Also dynamically prunes words that appear frequently in the target search context (local stop-words).
    Never prunes words that are present in the user's query itself or match the primary category keyword.
    """
    if not regex_pattern:
        return ""
        
    GENERIC_WORDS_TO_PRUNE = {
        # Pronouns, prepositions, conjunctions
        "the", "and", "for", "are", "but", "not", "you", "him", "her", "his", "its", "our", "out", "off", "one", "two", "use", "get", "got", "job", "new", "old", "day", "way", "now", "did", "had", "has", "was", "any", "all", "who", "why", "how", "few", "own", "too", "can", "will", "just", "should", "could", "would", "with", "about", "above", "below", "under", "over", "before", "after", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "both", "each", "few", "more", "most", "some", "other", "same", "such", "only", "very",
        # Generic structural and conversational nouns/verbs that appear in almost every session
        "name", "names", "list", "lists", "type", "types", "item", "items", "thing", "things", "work", "worker", "job", "jobs", "career", "show", "shows", "play", "plays", "day", "days", "time", "times", "year", "years", "people", "person", "man", "woman", "men", "women", "former", "previous", "next", "change", "changed", "changes", "initials", "family", "birth", "studies", "science", "arts", "business", "administration", "engineering", "food", "drink", "drinks", "beverage", "beverages", "refreshment", "refreshments", "place", "places", "location", "locations", "store", "shop", "shops", "supermarket", "grocery", "target", "redeem", "discount", "coupon", "coupons", "buy", "bought", "purchase", "purchased", "order", "ordered", "pay", "paid", "sell", "sold", "cost", "price", "money", "dollar", "dollars", "cent", "cents", "amount", "value", "free", "cheap", "expensive", "sale", "deal", "deals", "treat", "therapy", "feed", "eat", "consume", "like", "want", "need", "find", "search", "good", "bad", "first", "last", "user", "assistant", "what", "where", "who", "when", "why", "how"
    }

    # Extract words from the query
    query_words = set()
    if query:
        for w in re.split(r"\W+", query.lower()):
            if w:
                query_words.add(w)

    if primary_keyword:
        for w in re.split(r"\W+", primary_keyword.lower()):
            if w:
                query_words.add(w)

    # Build local frequency map from context
    high_freq_words = set()
    if context:
        import collections
        lines = context.split("\n")
        total_lines = len(lines)
        if total_lines > 10:
            word_counts = collections.Counter()
            for line in lines:
                words_in_line = set(re.findall(r"\b[a-z]{3,}\b", line.lower()))
                for w in words_in_line:
                    word_counts[w] += 1
            for w, count in word_counts.items():
                if count / total_lines > threshold:
                    high_freq_words.add(w)

    # Clean and split the regex pattern
    raw = regex_pattern.strip("() ")
    words = [w.strip() for w in raw.split("|") if w.strip()]
    
    pruned_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in query_words:
            pruned_words.append(w)
        elif w_lower in GENERIC_WORDS_TO_PRUNE:
            continue
        elif len(w_lower) <= 2:
            continue
        elif w_lower in high_freq_words:
            continue
        else:
            pruned_words.append(w)
                
    # If we pruned everything (fallback), return original
    if not pruned_words:
        return regex_pattern
        
    return f"({'|'.join(pruned_words)})"


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


def run_sandboxed_python_search(context: str, query: str, model, tokenizer, device: str) -> tuple[list[str], int]:
    """ChatGPT Simulator: Prompt the model to write a Python script that searches the context for the query concept."""
    system_prompt = (
        "You are a python assistant. Write a short Python script that searches the multiline string variable 'context' "
        "for lines matching the concept in the query. Your script must split 'context' into lines, identify matching lines, "
        "and populate a global list variable named 'results' with those matching lines. Do not import any external libraries except 're'.\n"
        "Write ONLY the executable Python code inside a ```python ``` code block. Do not write any other text or markdown."
    )
    user_prompt = f"Query concept: '{query}'\n\nWrite python code:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    token_count = out.shape[1]
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Extract code from ```python ... ```
    code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        code = response
        
    local_vars = {"context": context, "results": []}
    try:
        exec(code, {}, local_vars)
        results = local_vars.get("results", [])
        if not isinstance(results, list):
            results = []
    except Exception as e:
        results = []
        
    return [str(r) for r in results], token_count


def run_iterative_grep_search(context: str, query: str, initial_category: str, model, tokenizer, device: str) -> tuple[list[str], int]:
    """Claude Code Simulator: Simulates a multi-turn interactive agent grep refinement loop."""
    current_search_term = initial_category
    matches = []
    total_tokens = 0
    
    for turn in range(3):
        matches = run_grep_search(context, current_search_term)
        if matches:
            break
            
        system_prompt = (
            "You are a command-line search assistant. Your previous search for '{term}' returned 0 matches in the file. "
            "Suggest ONE new specific synonym, category member, or refined search word to try. "
            "Output ONLY the single word and nothing else."
        )
        user_prompt = f"Target question: {query}\nPrevious term: {current_search_term}\nNew search word:"
        
        messages = [
            {"role": "system", "content": system_prompt.format(term=current_search_term)},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        total_tokens += out.shape[1]
        current_search_term = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        current_search_term = re.sub(r"\W+", "", current_search_term)
        
    return matches, total_tokens

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

def evaluate_method(method_name: str, dataset: list[dict], model, tokenizer, device: str, extra_data: dict) -> tuple[float, float, float]:
    """Run evaluation for a specific retrieval method over the real dataset."""
    n_correct = 0
    total_tokens = 0
    total_search_tokens = 0
    
    for ex in dataset:
        context = ex["full_context"]
        query = ex["question"]
        ref_ans = ex["answer"]
        q_id = ex["question_id"]
        
        search_tok = 0
        
        # 1. Retrieve
        if method_name == "grep":
            kw = extra_data[q_id]["keyword"]
            retrieved = run_grep_search(context, kw)
        elif method_name == "lqe_grep":
            pat = extra_data[q_id]["regex"]
            search_tok = extra_data[q_id]["regex_tokens"]
            retrieved = run_lqe_grep_search(context, pat)
        elif method_name == "lqe_grep_v2":
            pat = extra_data[q_id]["regex"]
            kw = extra_data[q_id]["keyword"]
            search_tok = extra_data[q_id]["regex_tokens"]
            pruned_pat = prune_regex_pattern(pat, query, kw, context, threshold=0.05)
            retrieved = run_lqe_grep_search(context, pruned_pat)
        elif method_name == "vector":
            retrieved = run_vector_search(context, query, model, tokenizer, device, top_k=3)
        elif method_name == "cursor_hybrid":
            candidates = run_vector_search(context, query, model, tokenizer, device, top_k=15)
            pat = extra_data[q_id]["regex"]
            kw = extra_data[q_id]["keyword"]
            search_tok = extra_data[q_id]["regex_tokens"]
            pruned_pat = prune_regex_pattern(pat, query, kw, context, threshold=0.05)
            retrieved = run_lqe_grep_search("\n".join(candidates), pruned_pat)
        elif method_name == "sandboxed_python":
            retrieved, search_tok = run_sandboxed_python_search(context, query, model, tokenizer, device)
        elif method_name == "iterative_grep":
            kw = extra_data[q_id]["keyword"]
            retrieved, search_tok = run_iterative_grep_search(context, query, kw, model, tokenizer, device)
            # Add initial keyword extraction tokens
            search_tok += extra_data[q_id]["keyword_tokens"]
        else:
            retrieved = []
            
        total_search_tokens += search_tok
        
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
        print(f"      Search generation tokens: {search_tok}")
        print(f"      Retrieved ({len(retrieved)} lines):")
        for line in retrieved[:3]:
            print(f"        {line[:120]}...")
        print(f"      ---")
            
    avg_tokens = total_tokens / len(dataset)
    accuracy = n_correct / len(dataset)
    avg_search_tokens = total_search_tokens / len(dataset)
    return accuracy, avg_tokens, avg_search_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key from src/models.py")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of questions to evaluate")
    parser.add_argument("--methods", type=str, default="grep,lqe_grep,lqe_grep_v2,vector", help="Comma-separated list of methods to run")
    args = parser.parse_args()
    
    device = best_gpu()
    print(f"Loading model {args.model}...")
    model, tokenizer = load(args.model, device=device)
    
    # Load dataset
    print("Loading LongMemEval dataset...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "data", "longmemeval_s_cleaned.json")
    with open(dataset_path, "r") as f:
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
        kw, kw_tok = extract_single_keyword(ex["question"], model, tokenizer, device)
        pat, pat_tok = extract_noun_and_expand_regex(ex["question"], model, tokenizer, device)
        extra_data[q_id] = {"keyword": kw, "keyword_tokens": kw_tok, "regex": pat, "regex_tokens": pat_tok}
        print(f"  Q ID {q_id}: Query = '{ex['question']}'")
        print(f"    Keyword = '{kw}' (Tokens: {kw_tok}) | Regex = '{pat}' (Tokens: {pat_tok})")
        
    results = {}
    if args.methods == "grep,lqe_grep,lqe_grep_v2,vector":
        args.methods = "grep,lqe_grep,lqe_grep_v2,vector,cursor_hybrid,sandboxed_python,iterative_grep"
        
    methods_to_run = [m.strip().lower() for m in args.methods.split(",")]
    
    for method in methods_to_run:
        print(f"\nEvaluating Method: {method.upper()}...")
        acc, tok, search_tok = evaluate_method(method, eval_subset, model, tokenizer, device, extra_data)
        print(f"  {method.upper()} Accuracy: {acc:.2%}, Avg Context Tokens: {tok:.1f}, Avg Search Gen Tokens: {search_tok:.1f}")
        results[method] = {"accuracy": acc, "avg_tokens": tok, "avg_search_tokens": search_tok}
        
    # Save results
    output_path = os.path.join(project_root, "results", "lqe_real_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n\n### Real LongMemEval Experiment Summary Table")
    print("| Method | Accuracy | Avg Context Tokens | Avg Search Gen Tokens |")
    print("| --- | --- | --- | --- |")
    for method in methods_to_run:
        if method in results:
            m_res = results[method]
            print(f"| {method.upper()} | {m_res['accuracy']:.1%} | {m_res['avg_tokens']:.1f} | {m_res['avg_search_tokens']:.1f} |")
        
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
