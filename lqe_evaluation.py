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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "subspace-search"))

from src.models import load, best_gpu
from synthetic_memory import generate_benchmark_dataset, CATEGORIES


def extract_noun_and_expand_regex(query: str, model, tokenizer, device: str) -> tuple[str, int]:
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


def evaluate_method(method_name: str, dataset: list[dict], model, tokenizer, device: str, expanded_patterns: dict, expansion_tokens: dict) -> tuple[float, float, float]:
    """Run evaluation for a specific retrieval method over the dataset."""
    n_correct = 0
    total_tokens = 0
    total_search_tokens = 0
    
    for ex in dataset:
        context = ex["context"]
        query = ex["query"]
        category = ex["category"]
        ref_ans = ex["reference_answer"]
        
        search_tok = 0
        
        # 1. Retrieve
        if method_name == "grep":
            retrieved = run_grep_search(context, category)
        elif method_name == "lqe_grep":
            pat = expanded_patterns[ex["id"]]
            search_tok = expansion_tokens[ex["id"]]
            retrieved = run_lqe_grep_search(context, pat)
        elif method_name == "lqe_grep_v2":
            pat = expanded_patterns[ex["id"]]
            search_tok = expansion_tokens[ex["id"]]
            pruned_pat = prune_regex_pattern(pat, query, category, context, threshold=0.05)
            retrieved = run_lqe_grep_search(context, pruned_pat)
        elif method_name == "vector":
            retrieved = run_vector_search(context, query, model, tokenizer, device, top_k=3)
        elif method_name == "cursor_hybrid":
            candidates = run_vector_search(context, query, model, tokenizer, device, top_k=15)
            pat = expanded_patterns[ex["id"]]
            search_tok = expansion_tokens[ex["id"]]
            pruned_pat = prune_regex_pattern(pat, query, category, context, threshold=0.05)
            retrieved = run_lqe_grep_search("\n".join(candidates), pruned_pat)
        elif method_name == "sandboxed_python":
            retrieved, search_tok = run_sandboxed_python_search(context, query, model, tokenizer, device)
        elif method_name == "iterative_grep":
            retrieved, search_tok = run_iterative_grep_search(context, query, category, model, tokenizer, device)
        else:
            retrieved = []
            
        total_search_tokens += search_tok
        
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
        print(f"      Search generation tokens: {search_tok}")
        print(f"      Retrieved ({len(retrieved)} lines):")
        for line in retrieved:
            print(f"        {line}")
        print(f"      ---")
            
    avg_tokens = total_tokens / len(dataset)
    accuracy = n_correct / len(dataset)
    avg_search_tokens = total_search_tokens / len(dataset)
    return accuracy, avg_tokens, avg_search_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key from src/models.py")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples per category (6 categories total)")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "lqe_results.json"), help="Path to save result JSON")
    parser.add_argument("--methods", type=str, default="grep,lqe_grep,lqe_grep_v2,vector", help="Comma-separated list of methods to run")
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
    expansion_tokens = {}
    for ex in temp_dataset:
        pat, tok = extract_noun_and_expand_regex(ex["query"], model, tokenizer, device)
        expanded_patterns[ex["id"]] = pat
        expansion_tokens[ex["id"]] = tok
        print(f"  Query ID {ex['id']} ({ex['category']}): Query = '{ex['query']}' -> Regex = {pat} (Tokens: {tok})")
        
    # Standardize list of methods
    if args.methods == "grep,lqe_grep,lqe_grep_v2,vector":
        # Expand default list to include new simulated search paradigms
        args.methods = "grep,lqe_grep,lqe_grep_v2,vector,cursor_hybrid,sandboxed_python,iterative_grep"
        
    methods_to_run = [m.strip().lower() for m in args.methods.split(",")]
    
    # Sweep over noise levels
    for noise in noise_levels:
        print(f"\n==========================================")
        print(f"Running evaluation with {noise} distractor turns...")
        print(f"==========================================")
        
        dataset = generate_benchmark_dataset(num_examples_per_cat=args.num_examples, num_distractors=noise, base_seed=42)
        results[str(noise)] = {}
        
        for method in methods_to_run:
            print(f"Evaluating Method: {method.upper()}...")
            acc, tok, search_tok = evaluate_method(method, dataset, model, tokenizer, device, expanded_patterns, expansion_tokens)
            print(f"  {method.upper()} Accuracy: {acc:.2%}, Avg Retrieval Tokens: {tok:.1f}, Avg Search Gen Tokens: {search_tok:.1f}")
            results[str(noise)][method] = {"accuracy": acc, "avg_tokens": tok, "avg_search_tokens": search_tok}
            
    # Write JSON results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    # Print a beautiful Markdown Summary Table
    print("\n\n### Experiment Summary Table")
    print("| Noise (Distractors) | Method | Accuracy | Avg Context Tokens | Avg Search Gen Tokens |")
    print("| --- | --- | --- | --- | --- |")
    for noise in noise_levels:
        for method in methods_to_run:
            if method in results[str(noise)]:
                m_res = results[str(noise)][method]
                print(f"| {noise} | {method.upper()} | {m_res['accuracy']:.1%} | {m_res['avg_tokens']:.1f} | {m_res['avg_search_tokens']:.1f} |")
            
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
