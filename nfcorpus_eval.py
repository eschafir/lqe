import argparse
import json
import os
import re
import sys
import random
import time
import torch
import torch.nn.functional as F

# Add project root to sys.path to find src.models
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import load, best_gpu

def extract_grep_keywords(query: str, model, tokenizer, device: str) -> list[str]:
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
        
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    words = response.lower().split()
    # clean words from punctuation
    cleaned = [w.strip(".,:;\"'()[]") for w in words if len(w) > 2]
    return cleaned[:2]

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

def expand_query_lqe(query: str, model, tokenizer, device: str) -> str:
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
        
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return clean_regex_pattern(response)


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
    corpus = {}
    with open("data/nfcorpus/corpus.jsonl", "r") as f:
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
    with open("data/nfcorpus/queries.jsonl", "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q
            
    # Load relevance labels (qrels)
    print("Loading relevance judgments (qrels)...")
    qrels = {}
    with open("data/nfcorpus/qrels/test.tsv", "r") as f:
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
    
    methods_to_run = [m.strip().lower() for m in args.methods.split(",")]
    
    grep_successes = 0
    lqe_successes = 0
    lqe_successes_v2 = 0
    vec_successes = 0
    
    grep_tokens = 0
    lqe_tokens = 0
    lqe_tokens_v2 = 0
    
    for idx, q_id in enumerate(selected_q_ids):
        query_text = queries[q_id]["text"]
        target_doc_ids = qrels[q_id]
        
        # Pick one target document as the needle
        target_id = target_doc_ids[0]
        target_doc = corpus[target_id]
        
        # Sample distractors
        all_doc_ids = list(corpus.keys())
        distractor_candidates = [d_id for d_id in all_doc_ids if d_id not in target_doc_ids]
        sampled_distractors = random.sample(distractor_candidates, args.haystack_size - 1)
        
        # Form haystack
        haystack = [target_doc] + [corpus[d_id] for d_id in sampled_distractors]
        random.shuffle(haystack)
        
        # Pre-extract terms
        keywords = extract_grep_keywords(query_text, model, tokenizer, device) if "grep" in methods_to_run or "lqe_grep_v2" in methods_to_run else []
        lqe_pattern = expand_query_lqe(query_text, model, tokenizer, device) if "lqe_grep" in methods_to_run or "lqe_grep_v2" in methods_to_run else ""
        lqe_pattern_v2 = prune_regex_pattern(lqe_pattern, query_text, " ".join(keywords), corpus_df, total_docs, threshold=0.05) if "lqe_grep_v2" in methods_to_run else ""
        
        # Run Searches
        grep_results = search_grep(haystack, keywords) if "grep" in methods_to_run else []
        lqe_results = search_lqe_grep(haystack, lqe_pattern) if "lqe_grep" in methods_to_run else []
        lqe_results_v2 = search_lqe_grep(haystack, lqe_pattern_v2) if "lqe_grep_v2" in methods_to_run else []
        vec_results = search_vector(haystack, query_text, model, tokenizer, device) if "vector" in methods_to_run else []
        
        # Compute successes (Success@3)
        g_suc = 1 if target_id in grep_results else 0
        l_suc = 1 if target_id in lqe_results else 0
        l_suc_v2 = 1 if target_id in lqe_results_v2 else 0
        v_suc = 1 if target_id in vec_results else 0
        
        grep_successes += g_suc
        lqe_successes += l_suc
        lqe_successes_v2 += l_suc_v2
        vec_successes += v_suc
        
        # Measure token footprint
        if "grep" in methods_to_run:
            grep_passages = []
            for doc in haystack:
                text = f"{doc['title']} {doc['text']}"
                for kw in keywords:
                    pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                    for line in text.split("\n"):
                        if pattern.search(line):
                            grep_passages.append(line)
            grep_tokens += len(tokenizer.tokenize("\n".join(grep_passages)))
        
        if "lqe_grep" in methods_to_run:
            lqe_passages = []
            if lqe_pattern.startswith("(") and lqe_pattern.endswith(")"):
                pattern_str = rf"\b{lqe_pattern}\b"
            else:
                pattern_str = rf"\b({lqe_pattern.strip('()')})\b"
            
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for doc in haystack:
                    text = f"{doc['title']} {doc['text']}"
                    for line in text.split("\n"):
                        if pattern.search(line):
                            lqe_passages.append(line)
            except re.error:
                pass
            lqe_tokens += len(tokenizer.tokenize("\n".join(lqe_passages)))
        
        if "lqe_grep_v2" in methods_to_run:
            lqe_passages_v2 = []
            if lqe_pattern_v2.startswith("(") and lqe_pattern_v2.endswith(")"):
                pattern_str_v2 = rf"\b{lqe_pattern_v2}\b"
            else:
                pattern_str_v2 = rf"\b({lqe_pattern_v2.strip('()')})\b"
            
            try:
                pattern_v2 = re.compile(pattern_str_v2, re.IGNORECASE)
                for doc in haystack:
                    text = f"{doc['title']} {doc['text']}"
                    for line in text.split("\n"):
                        if pattern_v2.search(line):
                            lqe_passages_v2.append(line)
            except re.error:
                pass
            lqe_tokens_v2 += len(tokenizer.tokenize("\n".join(lqe_passages_v2)))
        
        print(f"  Query {idx+1}/{len(selected_q_ids)}: '{query_text}'")
        if "grep" in methods_to_run or "lqe_grep_v2" in methods_to_run:
            print(f"    Keywords: {keywords}")
        if "lqe_grep" in methods_to_run:
            print(f"    LQE: {lqe_pattern}")
        if "lqe_grep_v2" in methods_to_run:
            print(f"    LQE v2: {lqe_pattern_v2}")
        print(f"    Target ID: {target_id}")
        if "grep" in methods_to_run:
            print(f"    Grep top-3: {grep_results} (Success: {g_suc})")
        if "lqe_grep" in methods_to_run:
            print(f"    LQE  top-3: {lqe_results} (Success: {l_suc})")
        if "lqe_grep_v2" in methods_to_run:
            print(f"    LQE2 top-3: {lqe_results_v2} (Success: {l_suc_v2})")
        if "vector" in methods_to_run:
            print(f"    Vector top-3: {vec_results} (Success: {v_suc})")
        print("-" * 50)
        
    print("\n" + "=" * 50)
    print("NFCorpus Retrieval Evaluation Summary")
    print("=" * 50)
    if "grep" in methods_to_run:
        print(f"Vanilla Grep  Success@3: {grep_successes / len(selected_q_ids):.2%} | Avg Tokens: {grep_tokens / len(selected_q_ids):.1f}")
    if "lqe_grep" in methods_to_run:
        print(f"LQE-Grep      Success@3: {lqe_successes / len(selected_q_ids):.2%} | Avg Tokens: {lqe_tokens / len(selected_q_ids):.1f}")
    if "lqe_grep_v2" in methods_to_run:
        print(f"LQE-Grep v2   Success@3: {lqe_successes_v2 / len(selected_q_ids):.2%} | Avg Tokens: {lqe_tokens_v2 / len(selected_q_ids):.1f}")
    if "vector" in methods_to_run:
        print(f"Vector Search Success@3: {vec_successes / len(selected_q_ids):.2%}")
        
    # Save results
    results = {}
    if "grep" in methods_to_run:
        results["grep"] = {"success@3": grep_successes / len(selected_q_ids), "avg_tokens": grep_tokens / len(selected_q_ids)}
    if "lqe_grep" in methods_to_run:
        results["lqe_grep"] = {"success@3": lqe_successes / len(selected_q_ids), "avg_tokens": lqe_tokens / len(selected_q_ids)}
    if "lqe_grep_v2" in methods_to_run:
        results["lqe_grep_v2"] = {"success@3": lqe_successes_v2 / len(selected_q_ids), "avg_tokens": lqe_tokens_v2 / len(selected_q_ids)}
    if "vector" in methods_to_run:
        results["vector"] = {"success@3": vec_successes / len(selected_q_ids)}
        
    with open("lqe_nfcorpus_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to lqe_nfcorpus_results.json")

if __name__ == "__main__":
    main()
