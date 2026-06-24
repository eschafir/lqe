import argparse
import json
import os
import re
import sys
import random
import collections
import torch
from rank_bm25 import BM25Okapi

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
    cleaned = [w.strip(".,:;\"'()[]") for w in words if len(w) > 2]
    return cleaned[:2], token_count

def clean_regex_pattern(raw_pattern: str) -> str:
    """Sanitize and format query expansion results to guarantee valid regex syntax."""
    cleaned = raw_pattern.strip("()[]\"' ")
    cleaned = cleaned.replace(" ", "").replace("\n", "")
    cleaned = re.sub(r"[^a-zA-Z0-9|_-]", "", cleaned)
    parts = [p for p in cleaned.split("|") if p]
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
    """LQE v2: Filter out generic, high-frequency, or short English words from the expanded regex."""
    if not regex_pattern:
        return ""
        
    GENERIC_WORDS_TO_PRUNE = {
        "the", "and", "for", "are", "but", "not", "you", "him", "her", "his", "its", "our", "out", "off", "one", "two", "use", "get", "got", "job", "new", "old", "day", "way", "now", "did", "had", "has", "was", "any", "all", "who", "why", "how", "few", "own", "too", "can", "will", "just", "should", "could", "would", "with", "about", "above", "below", "under", "over", "before", "after", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "both", "each", "few", "more", "most", "some", "other", "same", "such", "only", "very",
        "name", "names", "list", "lists", "type", "types", "item", "items", "thing", "things", "work", "worker", "job", "jobs", "career", "show", "shows", "play", "plays", "day", "days", "time", "times", "year", "years", "people", "person", "man", "woman", "men", "women", "former", "previous", "next", "change", "changed", "changes", "initials", "family", "birth", "studies", "science", "arts", "business", "administration", "engineering", "food", "drink", "drinks", "beverage", "beverages", "refreshment", "refreshments", "place", "places", "location", "locations", "store", "shop", "shops", "supermarket", "grocery", "target", "redeem", "discount", "coupon", "coupons", "buy", "bought", "purchase", "purchased", "order", "ordered", "pay", "paid", "sell", "sold", "cost", "price", "money", "dollar", "dollars", "cent", "cents", "amount", "value", "free", "cheap", "expensive", "sale", "deal", "deals", "treat", "therapy", "feed", "eat", "consume", "like", "want", "need", "find", "search", "good", "bad", "first", "last", "user", "assistant", "what", "where", "who", "when", "why", "how"
    }

    query_words = set()
    if query:
        for w in re.split(r"\W+", query.lower()):
            if w:
                query_words.add(w)

    if primary_keyword:
        for w in re.split(r"\W+", primary_keyword.lower()):
            if w:
                query_words.add(w)

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
                
    if not pruned_words:
        return regex_pattern
        
    return f"({'|'.join(pruned_words)})"

def search_grep(haystack: list[dict], keywords: list[str]) -> list[str]:
    """Search haystack using exact keywords. Rank documents by number of matches."""
    scores = []
    for doc in haystack:
        count = 0
        text = f"{doc['title']} {doc['text']}"
        for kw in keywords:
            pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            count += len(pattern.findall(text))
        scores.append((doc["_id"], count))
    
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
        return [doc["_id"] for doc in haystack[:3]]
        
    for doc in haystack:
        text = f"{doc['title']} {doc['text']}"
        count = len(pattern.findall(text))
        scores.append((doc["_id"], count))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:3]]

def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return [w.strip() for w in re.split(r"\W+", text.lower()) if w.strip()]

def search_bm25(haystack: list[dict], query: str) -> list[str]:
    """Search haystack using rank_bm25."""
    corpus_tokens = [tokenize(f"{doc['title']} {doc['text']}") for doc in haystack]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(haystack)), key=lambda i: scores[i], reverse=True)
    return [haystack[i]["_id"] for i in ranked_indices[:3]]

def search_mugi(haystack: list[dict], query: str, gen_cands: list[str]) -> list[str]:
    """Search haystack using MuGI query expansion logic."""
    if not gen_cands:
        return search_bm25(haystack, query)
    gen_ref = ' '.join(gen_cands[:5])
    adaptive_times = 6
    if len(query) > 0:
        times = (len(gen_ref) // len(query)) // adaptive_times
    else:
        times = 1
    enhanced_query = (query + ' ') * times + gen_ref
    return search_bm25(haystack, enhanced_query)

def generate_mugi_references_qwen(query: str, model, tokenizer, device: str, num_docs: int = 5) -> tuple[list[str], int]:
    """Generate pseudo-references using Qwen-2.5-1.5B for MuGI query expansion."""
    messages = [
        {
            "role": "system",
            "content": "You are PassageGenGPT, an AI capable of generating concise, informative, and clear pseudo passages on specific topics."
        },
        {
            "role": "user",
            "content": f"Generate one passage that is relevant to the following query: '{query}'. The passage should be concise, informative, and clear"
        },
        {
            "role": "assistant",
            "content": "Sure, here's a passage relevant to the query:"
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    references = []
    total_tokens = 0
    
    for _ in range(num_docs):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        total_tokens += out.shape[1]
        response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        references.append(response)
        
    return references, total_tokens

def expand_query_lqe_coordinated(query: str, model, tokenizer, device: str) -> tuple[list[str], int]:
    """Prompt the model to group the main query concepts into distinct lists of synonyms separated by &&."""
    system_prompt = (
        "You are a precise search assistant. Your job is to perform multi-concept query expansion. "
        "Identify 2 or 3 distinct semantic concepts/nouns in the user's query. Expand each concept into a "
        "pipe-separated list of synonyms and related terms enclosed in parentheses. "
        "Join the concept groups using '&&' without spaces. Do not use common words like 'and', 'the'.\n\n"
        "Example:\n"
        "Query: \"Breast Cancer Cells Feed on Cholesterol\"\n"
        "Output: (breast|cancer|carcinoma|tumor|malignancy)&&(cholesterol|lipid|fat|statins)&&(cells|feed|eat|consume)\n\n"
        "Respond ONLY with the final expanded concepts joined by '&&'. Do not output any other text."
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
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    token_count = out.shape[1]
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    parts = response.split("&&")
    cleaned_parts = [clean_regex_pattern(p) for p in parts if p.strip()]
    return cleaned_parts, token_count

def search_lqe_grep_weighted(haystack: list[dict], regex_pattern: str, corpus_df: dict, total_docs: int) -> list[str]:
    """Search haystack using IDF-weighted matches."""
    import math
    raw = regex_pattern.strip("() ")
    terms = [w.strip() for w in raw.split("|") if w.strip()]
    
    weights = {}
    for t in terms:
        t_lower = t.lower()
        df = corpus_df.get(t_lower, 0)
        weights[t_lower] = math.log(1.0 + total_docs / max(df, 1))
        
    scores = []
    for doc in haystack:
        score = 0.0
        text = f"{doc['title']} {doc['text']}".lower()
        for t in terms:
            t_lower = t.lower()
            pattern = re.compile(rf"\b{re.escape(t_lower)}\b", re.IGNORECASE)
            count = len(pattern.findall(text))
            score += count * weights[t_lower]
        scores.append((doc["_id"], score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:3]]

def search_lqe_grep_stemmed(haystack: list[dict], regex_pattern: str) -> list[str]:
    """Search haystack using suffix-stripped wildcard patterns."""
    raw = regex_pattern.strip("() ")
    terms = [w.strip() for w in raw.split("|") if w.strip()]
    
    stemmed_parts = []
    suffixes = ("s", "es", "ed", "ing", "ly", "y")
    for t in terms:
        stemmed = t
        for suffix in suffixes:
            if t.lower().endswith(suffix) and len(t) - len(suffix) > 3:
                stemmed = t[:-len(suffix)]
                break
        stemmed_parts.append(rf"{re.escape(stemmed)}\w*")
        
    stemmed_regex = f"({'|'.join(stemmed_parts)})"
    return search_lqe_grep(haystack, stemmed_regex)

def search_lqe_grep_coordinated(haystack: list[dict], concept_regexes: list[str]) -> list[str]:
    """Search haystack enforcing concept group coordination."""
    patterns = []
    for r in concept_regexes:
        r_clean = clean_regex_pattern(r)
        if r_clean.startswith("(") and r_clean.endswith(")"):
            pat_str = rf"\b{r_clean}\b"
        else:
            pat_str = rf"\b({r_clean.strip('()')})\b"
        try:
            patterns.append(re.compile(pat_str, re.IGNORECASE))
        except re.error:
            pass
            
    scores = []
    for doc in haystack:
        text = f"{doc['title']} {doc['text']}"
        concept_matches = 0
        total_matches = 0
        for pat in patterns:
            matches = len(pat.findall(text))
            if matches > 0:
                concept_matches += 1
                total_matches += matches
        score = concept_matches * 1000 + total_matches
        scores.append((doc["_id"], score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:3]]

def stem_word(w: str) -> str:
    """Helper to strip common English suffixes to get word stems."""
    suffixes = ("s", "es", "ed", "ing", "ly", "y")
    w_lower = w.lower()
    for suffix in suffixes:
        if w_lower.endswith(suffix) and len(w_lower) - len(suffix) > 3:
            return w_lower[:-len(suffix)]
    return w_lower

def search_lqe_grep_v4(haystack: list[dict], regex_pattern: str, corpus_stem_df: dict, total_docs: int) -> list[str]:
    """LQE-Grep v4: Suffix-stripped wildcard stemming combined with stem-based IDF weights."""
    import math
    raw = regex_pattern.strip("() ")
    terms = [w.strip() for w in raw.split("|") if w.strip()]
    
    stems = [stem_word(t) for t in terms]
    unique_stems = list(set(stems))
    
    weights = {}
    for s in unique_stems:
        df = corpus_stem_df.get(s, 0)
        weights[s] = math.log(1.0 + total_docs / max(df, 1))
        
    scores = []
    for doc in haystack:
        score = 0.0
        text = f"{doc['title']} {doc['text']}".lower()
        for s in unique_stems:
            # Match stem and any characters following it
            pattern = re.compile(rf"\b{re.escape(s)}\w*", re.IGNORECASE)
            count = len(pattern.findall(text))
            score += count * weights[s]
        scores.append((doc["_id"], score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:3]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key from src/models.py")
    parser.add_argument("--num-queries", type=int, default=323, help="Number of queries to evaluate")
    parser.add_argument("--haystack-size", type=int, default=50, help="Number of documents in each haystack")
    parser.add_argument("--methods", type=str, default="grep,lqe_grep,lqe_grep_v2,bm25,mugi_gpt4,mugi_gpt35,mugi_qwen,lqe_grep_v3_stemmed,lqe_grep_v3_weighted,lqe_grep_v3_coordinated,lqe_grep_v4", help="Comma-separated list of methods to run")
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
    corpus_df = collections.Counter()
    corpus_stem_df = collections.Counter()
    total_docs = len(corpus)
    for doc in corpus.values():
        text = f"{doc['title']} {doc['text']}".lower()
        words = set(re.findall(r"\b[a-z]{3,}\b", text))
        for w in words:
            corpus_df[w] += 1
            corpus_stem_df[stem_word(w)] += 1
            
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
        f.readline()  # Skip header
        for line in f:
            q_id, doc_id, score = line.strip().split("\t")
            if int(score) >= 2:  # Keep highly relevant mappings
                if q_id not in qrels:
                    qrels[q_id] = []
                qrels[q_id].append(doc_id)
                
    # Load MuGI precomputed data
    print("Loading MuGI precomputed data...")
    mugi_data = {}
    mugi_refine_path = os.path.join(project_root, "Retrieval_MuGI", "exp", "gpt", "nfc_bm25_refine.json")
    if os.path.exists(mugi_refine_path):
        with open(mugi_refine_path, "r") as f:
            mugi_list = json.load(f)
            for item in mugi_list:
                mugi_data[item["query"].strip().lower()] = {
                    "gen_cand_gpt4": item.get("gen_cand_gpt4", []),
                    "gen_cand_gpt35": item.get("gen_cand_gpt35", [])
                }
        print(f"Loaded {len(mugi_data)} MuGI query expansions.")
    else:
        print("WARNING: MuGI precomputed data not found!")

    # Filter queries that have relevance labels
    valid_q_ids = [q_id for q_id in queries.keys() if q_id in qrels]
    random.seed(42)
    selected_q_ids = random.sample(valid_q_ids, min(args.num_queries, len(valid_q_ids)))
    
    print(f"\nEvaluating on {len(selected_q_ids)} test queries. Haystack size = {args.haystack_size}")
    
    methods_to_run = [m.strip().lower() for m in args.methods.split(",")]
    
    # Initialize trackers
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
        distractor_candidates = [d_id for d_id in all_doc_ids if d_id not in target_doc_ids]
        sampled_distractors = random.sample(distractor_candidates, args.haystack_size - 1)
        haystack = [target_doc] + [corpus[d_id] for d_id in sampled_distractors]
        random.shuffle(haystack)
        
        # Generate regex/keywords and track tokens
        keywords, kw_tok = extract_grep_keywords(query_text, model, tokenizer, device)
        lqe_pattern, pat_tok = expand_query_lqe(query_text, model, tokenizer, device)
        lqe_pattern_v2 = prune_regex_pattern(lqe_pattern, query_text, " ".join(keywords), corpus_df, total_docs, threshold=0.05)
        
        # Load MuGI references
        mugi_entry = mugi_data.get(query_text.strip().lower(), {})
        gpt4_ref = mugi_entry.get("gen_cand_gpt4", [])
        gpt35_ref = mugi_entry.get("gen_cand_gpt35", [])
        
        # Token count for MuGI pseudo references (search generation tokens)
        gpt4_ref_tok = len(tokenizer.tokenize(' '.join(gpt4_ref[:5]))) if gpt4_ref else 0
        gpt35_ref_tok = len(tokenizer.tokenize(' '.join(gpt35_ref[:5]))) if gpt35_ref else 0
        
        # Generate MuGI references using Qwen
        qwen_ref, qwen_ref_tok = generate_mugi_references_qwen(query_text, model, tokenizer, device, num_docs=5)
        
        # Generate coordinated regex concepts
        concepts, co_tok = expand_query_lqe_coordinated(query_text, model, tokenizer, device)
        
        print(f"\nQuery {idx+1}/{len(selected_q_ids)}: '{query_text}'")
        
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
            elif method == "lqe_grep_v3_stemmed":
                res = search_lqe_grep_stemmed(haystack, lqe_pattern_v2)
                search_tok = pat_tok
            elif method == "lqe_grep_v3_weighted":
                res = search_lqe_grep_weighted(haystack, lqe_pattern_v2, corpus_df, total_docs)
                search_tok = pat_tok
            elif method == "lqe_grep_v3_coordinated":
                res = search_lqe_grep_coordinated(haystack, concepts)
                search_tok = co_tok
            elif method == "bm25":
                res = search_bm25(haystack, query_text)
            elif method == "mugi_gpt4":
                res = search_mugi(haystack, query_text, gpt4_ref)
                search_tok = gpt4_ref_tok
            elif method == "mugi_gpt35":
                res = search_mugi(haystack, query_text, gpt35_ref)
                search_tok = gpt35_ref_tok
            elif method == "mugi_qwen":
                res = search_mugi(haystack, query_text, qwen_ref)
                search_tok = qwen_ref_tok
            elif method == "lqe_grep_v4":
                res = search_lqe_grep_v4(haystack, lqe_pattern_v2, corpus_stem_df, total_docs)
                search_tok = pat_tok
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
            elif method in ["lqe_grep", "lqe_grep_v2", "lqe_grep_v3_weighted"]:
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
            elif method in ["lqe_grep_v3_stemmed", "lqe_grep_v4"]:
                raw = lqe_pattern_v2.strip("() ")
                terms = [w.strip() for w in raw.split("|") if w.strip()]
                stemmed_parts = []
                for t in terms:
                    s = stem_word(t)
                    stemmed_parts.append(rf"{re.escape(s)}\w*")
                stemmed_regex = f"({'|'.join(stemmed_parts)})"
                if stemmed_regex.startswith("(") and stemmed_regex.endswith(")"):
                    pattern_str = rf"\b{stemmed_regex}\b"
                else:
                    pattern_str = rf"\b({stemmed_regex.strip('()')})\b"
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    for doc in haystack:
                        text = f"{doc['title']} {doc['text']}"
                        for line in text.split("\n"):
                            if pattern.search(line):
                                passages.append(line)
                except re.error:
                    pass
            elif method == "lqe_grep_v3_coordinated":
                concept_patterns = []
                for r in concepts:
                    r_clean = clean_regex_pattern(r)
                    if r_clean.startswith("(") and r_clean.endswith(")"):
                        pat_str = rf"\b{r_clean}\b"
                    else:
                        pat_str = rf"\b({r_clean.strip('()')})\b"
                    try:
                        concept_patterns.append(re.compile(pat_str, re.IGNORECASE))
                    except re.error:
                        pass
                for doc in haystack:
                    text = f"{doc['title']} {doc['text']}"
                    for line in text.split("\n"):
                        matched = False
                        for pat in concept_patterns:
                            if pat.search(line):
                                matched = True
                                break
                        if matched:
                            passages.append(line)
            elif method in ["bm25", "mugi_gpt4", "mugi_gpt35", "mugi_qwen"]:
                # BM25 / MuGI return top-3 documents, so their entire texts are loaded into context
                for doc_id in res:
                    for doc in haystack:
                        if doc["_id"] == doc_id:
                            passages.append(f"{doc['title']} {doc['text']}")
                            
            tok_count = len(tokenizer.tokenize("\n".join(passages)))
            tokens[method] += tok_count
            print(f"    {method.upper()} Success: {is_success} | Context Tokens: {tok_count} | Search Gen Tokens: {search_tok}")
            
    print("\n" + "=" * 50)
    print("NFCorpus Retrieval Evaluation Summary (LQE vs MuGI)")
    print("=" * 50)
    
    n_queries = len(selected_q_ids)
    results = {}
    for method in methods_to_run:
        acc = successes[method] / n_queries
        avg_tok = tokens[method] / n_queries
        avg_search_tok = search_tokens[method] / n_queries
        print(f"{method.upper():16} Success@3: {acc:.2%} | Avg Context Tokens: {avg_tok:.1f} | Avg Search Gen Tokens: {avg_search_tok:.1f}")
        results[method] = {"success_at_3": acc, "avg_tokens": avg_tok, "avg_search_tokens": avg_search_tok}
        
    output_path = os.path.join(project_root, "results", "mugi_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
