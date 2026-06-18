import argparse
import json
import os
import re
import sys
import random
import time
import torch
import torch.nn.functional as F

# Add project root parent to sys.path to find src.models
project_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_parent)
sys.path.insert(0, os.path.join(project_parent, "subspace-search"))

from src.models import load, best_gpu

def extract_grep_keywords(query: str, model, tokenizer, device: str) -> tuple[list[str], int]:
    """Extract 1 or 2 primary keywords from the query for Vanilla Grep."""
    system_prompt = (
        "You are a precise search assistant. Your job is to extract 1 or 2 primary, highly specific nouns or terms "
        "from the user's search query to be used in a keyword search. "
        "Do not output any other text or punctuation. Output only the words separated by space in lowercase.\n\n"
        "Examples:\n"
        "Query: \"Breast Cancer Cells Feed on Cholesterol\"\n"
        "Output: cholesterol breast\n\n"
        "Query: \"Using Diet to Treat Asthma and Eczema\"\n"
        "Output: asthma eczema\n\n"
        "Query: \"Is organic food healthier?\"\n"
        "Output: organic food"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: \"{query}\"\nOutput:"}
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
    words = response.lower().split()
    # clean words from punctuation
    cleaned = [w.strip(".,:;\"'()[]") for w in words if len(w) > 2]
    return cleaned[:2], token_count

def clean_regex_pattern(raw_pattern: str) -> str:
    """Sanitize and format query expansion results to guarantee valid regex syntax."""
    # Strip any outer parentheses, brackets, or quotes
    cleaned = raw_pattern.strip("()[]\"' ")
    # Remove whitespace and newlines
    cleaned = cleaned.replace(" ", "").replace("\n", "")
    # Remove any characters that are not alphanumeric, pipe, hyphen, underscore
    cleaned = re.sub(r"[^a-zA-Z0-9|_-]", "", cleaned)
    # Split by pipe and filter out empty elements
    parts = [p for p in cleaned.split("|") if p]
    # Return formatted pattern
    return f"({'|'.join(parts)})"

def expand_query_lqe(query: str, model, tokenizer, device: str) -> tuple[str, int]:
    """Prompt the model to expand the main concepts in the query into a regex pattern of synonyms and related terms."""
    system_prompt = (
        "You are a precise search assistant. Your job is to perform query expansion. "
        "Extract the main concepts/nouns from the user's query and expand them into an exhaustive list of synonyms, "
        "scientific terms, and related terms, formatted as a pipe-separated regular expression (without spaces) "
        "enclosed in parentheses.\n\n"
        "Be extremely broad to ensure high search recall, but avoid very common words like 'and', 'the', 'is', 'for'.\n\n"
        "Examples:\n"
        "Query: \"Breast Cancer Cells Feed on Cholesterol\"\n"
        "Output: (breast|cancer|carcinoma|tumor|malignancy|cells|feed|eat|consume|nutrition|cholesterol|lipid|fat|statins|statin)\n\n"
        "Query: \"Using Diet to Treat Asthma and Eczema\"\n"
        "Output: (diet|nutrition|food|treat|therapy|asthma|eczema|allergy|respiratory|dermatitis|skin|lungs)\n\n"
        "Respond ONLY with the final pipe-separated regex pattern enclosed in parentheses, like (term1|term2|term3). "
        "Do not output any other text."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: \"{query}\"\nOutput:"}
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
    return clean_regex_pattern(response), token_count


def prune_regex_pattern(regex_pattern: str, query: str = "", primary_keyword: str = "", corpus_df: dict = None, total_docs: int = 1, threshold: float = 0.05) -> str:
    """
    LQE v2: Filter out generic, high-frequency, or short English words from the expanded regex.
    Also prunes words that appear in more than threshold% of the documents in the corpus.
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
        elif corpus_df and (corpus_df.get(w_lower, 0) / total_docs) > threshold:
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

def search_grep(haystack: list[dict], keywords: list[str]) -> list[str]:
    """Search haystack using exact keywords. Rank documents by number of matches."""
    scores = []
    for doc in haystack:
        count = 0
        text = f"{doc['title']} {doc['text']}"
        for kw in keywords:
            # Case insensitive match with word boundaries
            pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            count += len(pattern.findall(text))
        scores.append((doc["_id"], count))
    
    # Sort by match count desc
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:3]]

def search_lqe_grep(haystack: list[dict], regex_pattern: str) -> list[str]:
    """Search haystack using expanded LQE regex. Rank documents by number of matches."""
    scores = []
    if regex_pattern.startswith("(") and regex_pattern.endswith(")"):
        pattern_str = rf"\b{regex_pattern}\b"
    else:
        pattern_str = rf"\b({regex_pattern.strip('()')})\b"
        
    try:
        pattern = re.compile(pattern_str, re.IGNORECASE)
    except re.error:
        # Fallback if pattern is broken
        return [doc["_id"] for doc in haystack[:3]]
        
    for doc in haystack:
        text = f"{doc['title']} {doc['text']}"
        count = len(pattern.findall(text))
        scores.append((doc["_id"], count))
        
    # Sort by match count desc
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:3]]

def search_vector(haystack: list[dict], query: str, model, tokenizer, device: str) -> list[str]:
    """Search haystack using cosine similarity of text embeddings."""
    query_emb = embed_text(query, model, tokenizer, device)
    
    scores = []
    for doc in haystack:
        doc_text = f"{doc['title']}: {doc['text']}"
        doc_emb = embed_text(doc_text[:800], model, tokenizer, device) # Truncated to first 800 chars to speed up
        sim = F.cosine_similarity(query_emb.unsqueeze(0), doc_emb.unsqueeze(0)).item()
        scores.append((doc["_id"], sim))
        
    # Sort by similarity desc
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:3]]


def run_sandboxed_python_search(haystack: list[dict], query: str, model, tokenizer, device: str) -> tuple[list[str], int]:
    """ChatGPT Simulator: Prompt the model to write a Python script that searches the documents in the haystack."""
    system_prompt = (
        "You are a python assistant. Write a short Python script that searches the list of dictionaries 'haystack' "
        "(where each dictionary has '_id', 'title', and 'text' keys) for documents matching the concept in the query. "
        "Your script must process the documents, identify matching documents, "
        "and populate a global list variable named 'results' with their '_id' strings. Do not import any external libraries except 're'.\n"
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
        
    local_vars = {"haystack": haystack, "results": []}
    try:
        exec(code, {}, local_vars)
        results = local_vars.get("results", [])
        if not isinstance(results, list):
            results = []
    except Exception as e:
        results = []
        
    return [str(r) for r in results[:3]], token_count


def run_iterative_grep_search(haystack: list[dict], query: str, initial_keywords: list[str], model, tokenizer, device: str) -> tuple[list[str], int]:
    """Claude Code Simulator: Simulates a multi-turn interactive agent grep refinement loop over documents."""
    current_keywords = list(initial_keywords)
    matches = []
    total_tokens = 0
    
    for turn in range(3):
        matches = search_grep(haystack, current_keywords)
        has_match = False
        for doc in haystack:
            text = f"{doc['title']} {doc['text']}"
            for kw in current_keywords:
                pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                if pattern.search(text):
                    has_match = True
                    break
            if has_match:
                break
                
        if has_match:
            break
            
        system_prompt = (
            "You are a command-line search assistant. Your previous search for '{terms}' returned 0 matches in the documents. "
            "Suggest ONE new specific synonym, category member, or refined search word to try. "
            "Output ONLY the single word and nothing else."
        )
        user_prompt = f"Target query: {query}\nPrevious terms: {', '.join(current_keywords)}\nNew search word:"
        
        messages = [
            {"role": "system", "content": system_prompt.format(terms=', '.join(current_keywords))},
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
        new_term = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        new_term = re.sub(r"\W+", "", new_term)
        if new_term:
            current_keywords = [new_term]
        else:
            break
            
    return search_grep(haystack, current_keywords), total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key from src/models.py")
    parser.add_argument("--num-queries", type=int, default=15, help="Number of queries to evaluate")
    parser.add_argument("--haystack-size", type=int, default=50, help="Number of documents in each haystack")
    parser.add_argument("--methods", type=str, default="grep,lqe_grep,lqe_grep_v2,vector", help="Comma-separated list of methods to run")
    args = parser.parse_args()
    
    device = best_gpu()
    print(f"Loading model {args.model} on {device}...")
    model, tokenizer = load(args.model, device=device)
    
    # Load corpus
    print("Loading corpus...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = {}
    corpus_path = os.path.join(project_root, "data", "nfcorpus", "corpus.jsonl")
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
            
    # Precompute document frequencies for corpus words (global stopword filtering)
    print("Computing corpus word frequencies...")
    import collections
    corpus_df = collections.Counter()
    total_docs = len(corpus)
    for doc in corpus.values():
        text = f"{doc['title']} {doc['text']}".lower()
        words = set(re.findall(r"\b[a-z]{3,}\b", text))
        for w in words:
            corpus_df[w] += 1
            
    # Load queries
    print("Loading queries...")
    queries = {}
    queries_path = os.path.join(project_root, "data", "nfcorpus", "queries.jsonl")
    with open(queries_path, "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q
            
    # Load relevance labels (qrels)
    print("Loading relevance judgments (qrels)...")
    qrels = {}
    qrels_path = os.path.join(project_root, "data", "nfcorpus", "qrels", "test.tsv")
    with open(qrels_path, "r") as f:
        f.readline() # Skip header
        for line in f:
            q_id, doc_id, score = line.strip().split("\t")
            if int(score) >= 2: # Keep highly relevant mappings
                if q_id not in qrels:
                    qrels[q_id] = []
                qrels[q_id].append(doc_id)
                
    # Filter queries that have relevance labels
    valid_q_ids = [q_id for q_id in queries.keys() if q_id in qrels]
    random.seed(42)
    selected_q_ids = random.sample(valid_q_ids, min(args.num_queries, len(valid_q_ids)))
    
    print(f"\nEvaluating on {len(selected_q_ids)} test queries. Haystack size = {args.haystack_size}")
    
    # Standardize list of methods
    if args.methods == "grep,lqe_grep,lqe_grep_v2,vector":
        args.methods = "grep,lqe_grep,lqe_grep_v2,vector,cursor_hybrid,sandboxed_python,iterative_grep"
        
    methods_to_run = [m.strip().lower() for m in args.methods.split(",")]
    
    # Initialize trackers
    results = {}
    successes = {m: 0 for m in methods_to_run}
    tokens = {m: 0 for m in methods_to_run}
    search_tokens = {m: 0 for m in methods_to_run}
    
    for idx, q_id in enumerate(selected_q_ids):
        query_text = queries[q_id]["text"]
        target_doc_ids = qrels[q_id]
        
        # Pick one target document as the needle
        target_id = target_doc_ids[0]
        target_doc = corpus[target_id]
        
        # Form haystack
        all_doc_ids = list(corpus.keys())
        if args.haystack_size <= 0 or args.haystack_size >= len(all_doc_ids):
            haystack = list(corpus.values())
        else:
            distractor_candidates = [d_id for d_id in all_doc_ids if d_id not in target_doc_ids]
            sampled_distractors = random.sample(distractor_candidates, args.haystack_size - 1)
            haystack = [target_doc] + [corpus[d_id] for d_id in sampled_distractors]
            random.shuffle(haystack)
        
        # Pre-extract terms and track search generation tokens
        keywords, kw_tok = extract_grep_keywords(query_text, model, tokenizer, device)
        lqe_pattern, pat_tok = expand_query_lqe(query_text, model, tokenizer, device)
        lqe_pattern_v2 = prune_regex_pattern(lqe_pattern, query_text, " ".join(keywords), corpus_df, total_docs, threshold=0.05)
        
        print(f"\nQuery {idx+1}/{len(selected_q_ids)}: '{query_text}'")
        print(f"  Keywords: {keywords} (Tokens: {kw_tok})")
        print(f"  LQE: {lqe_pattern} (Tokens: {pat_tok})")
        print(f"  Target ID: {target_id}")
        
        for method in methods_to_run:
            search_tok = 0
            
            if method == "grep":
                res = search_grep(haystack, keywords)
            elif method == "lqe_grep":
                res = search_lqe_grep(haystack, lqe_pattern)
                search_tok = pat_tok
            elif method == "lqe_grep_v2":
                res = search_lqe_grep(haystack, lqe_pattern_v2)
                search_tok = pat_tok
            elif method == "vector":
                res = search_vector(haystack, query_text, model, tokenizer, device)
            elif method == "cursor_hybrid":
                candidates = search_vector(haystack, query_text, model, tokenizer, device)[:15]
                # Filter haystack
                cand_docs = [doc for doc in haystack if doc["_id"] in candidates]
                res = search_lqe_grep(cand_docs, lqe_pattern_v2)
                search_tok = pat_tok
            elif method == "sandboxed_python":
                res, search_tok = run_sandboxed_python_search(haystack, query_text, model, tokenizer, device)
            elif method == "iterative_grep":
                res, search_tok = run_iterative_grep_search(haystack, query_text, keywords, model, tokenizer, device)
                search_tok += kw_tok
            else:
                res = []
                
            is_success = 1 if target_id in res else 0
            successes[method] += is_success
            search_tokens[method] += search_tok
            
            # Measure context token footprint
            passages = []
            if method == "grep":
                for doc in haystack:
                    text = f"{doc['title']} {doc['text']}"
                    for kw in keywords:
                        pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                        for line in text.split("\n"):
                            if pattern.search(line):
                                passages.append(line)
            elif method in ["lqe_grep", "lqe_grep_v2", "cursor_hybrid"]:
                pat_to_use = lqe_pattern if method == "lqe_grep" else lqe_pattern_v2
                if pat_to_use.startswith("(") and pat_to_use.endswith(")"):
                    pattern_str = rf"\b{pat_to_use}\b"
                else:
                    pattern_str = rf"\b({pat_to_use.strip('()')})\b"
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    for doc in haystack:
                        text = f"{doc['title']} {doc['text']}"
                        for line in text.split("\n"):
                            if pattern.search(line):
                                passages.append(line)
                except re.error:
                    pass
            elif method == "vector":
                # Vector returns top-3 documents, so the entire top-3 documents are matched
                for doc_id in res:
                    for doc in haystack:
                        if doc["_id"] == doc_id:
                            passages.append(f"{doc['title']} {doc['text']}")
            elif method in ["sandboxed_python", "iterative_grep"]:
                # Simply count the retrieved document lines
                for doc_id in res:
                    for doc in haystack:
                        if doc["_id"] == doc_id:
                            passages.append(f"{doc['title']} {doc['text']}")
                            
            tok_count = len(tokenizer.tokenize("\n".join(passages)))
            tokens[method] += tok_count
            print(f"    {method.upper()} Success: {is_success} | Context Tokens: {tok_count} | Search Gen Tokens: {search_tok}")
            
        print("-" * 50)
        
    print("\n" + "=" * 50)
    print("NFCorpus Retrieval Evaluation Summary")
    print("=" * 50)
    
    n_queries = len(selected_q_ids)
    for method in methods_to_run:
        acc = successes[method] / n_queries
        avg_tok = tokens[method] / n_queries
        avg_search_tok = search_tokens[method] / n_queries
        print(f"{method.upper():16} Success@3: {acc:.2%} | Avg Context Tokens: {avg_tok:.1f} | Avg Search Gen Tokens: {avg_search_tok:.1f}")
        results[method] = {"success_at_3": acc, "avg_tokens": avg_tok, "avg_search_tokens": avg_search_tok}
        
    # Save results
    output_path = os.path.join(project_root, "results", "lqe_nfcorpus_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
