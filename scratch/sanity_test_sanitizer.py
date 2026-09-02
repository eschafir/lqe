import os
import sys
import json
import argparse
import re
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models import load, best_gpu

stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'could', 'didn', 'doesn', 'hadn',
    'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])

def clean_query_current(text, domain=None):
    """Current sanitizer used in evaluate_hybrid.py: strips all non-alphanumeric chars."""
    text = re.sub(r'<[^>]+>', '', text)
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

def clean_query_raw(text, domain=None):
    """Raw query: only unescape HTML and strip tags, keep all syntax and punctuation intact."""
    text = (text.replace('&lt;', '<').replace('&gt;', '>')
            .replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'"))
    text = re.sub(r'<[^>]+>', ' ', text)
    return ' '.join(text.split())

def clean_query_smart(text, domain=None):
    """Syntax-preserving domain-aware sanitizer."""
    text = (text.replace('&lt;', '<').replace('&gt;', '>')
            .replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'"))
    text = re.sub(r'<[^>]+>', ' ', text)
    words = text.split()
    kept = []
    
    if domain == 'math':
        for w in words:
            # Preserve anything with math/LaTeX notation
            if re.search(r'[$_\\^\{\}=\+\-\*/<>\(\)\[\]]', w):
                kept.append(w)
            else:
                core = re.sub(r'^\W+|\W+$', '', w).lower()
                # Keep single letters (math variables like x, y, n)
                if core in stop_words and len(core) > 1:
                    continue
                kept.append(w)
    elif domain == 'askubuntu':
        negations = {'not', 'no', 'fail', 'error', 'cannot', 'cant', 'unable'}
        for w in words:
            # Preserve CLI commands, package names, flags, file paths
            if re.search(r'[-_/\.~:]', w):
                kept.append(w)
            else:
                core = re.sub(r'^\W+|\W+$', '', w).lower()
                if core in negations:
                    kept.append(w)
                elif core in stop_words:
                    continue
                else:
                    kept.append(w)
    else:
        for w in words:
            core = re.sub(r'^\W+|\W+$', '', w).lower()
            if core in stop_words and w.lower() == core:
                continue
            kept.append(w)
            
    return ' '.join(kept)

SANITIZERS = {
    "current": clean_query_current,
    "raw": clean_query_raw,
    "smart": clean_query_smart,
}

def min_max_normalize(scores):
    s_min = scores.min()
    s_max = scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return torch.zeros_like(scores)

def main():
    parser = argparse.ArgumentParser(description="Sanity test query sanitization on math & askubuntu")
    parser.add_argument("--checkpoint-path", type=str, default="results/vsplade_student_checkpoint_7b.pt")
    parser.add_argument("--model", type=str, default="qwen-7b")
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--targets-path", type=str, default="results/hive_new_test_targets.json")
    parser.add_argument("--splits", type=str, default="math,askubuntu")
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--use-prf", action="store_true", default=True)
    parser.add_argument("--prf-k", type=int, default=3)
    parser.add_argument("--prf-alpha", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    args = parser.parse_args()

    # Determine device with best GPU
    device = best_gpu()
    print(f"Loading Student model '{args.model}' on {device}...")
    model, tokenizer = load(args.model, device=device)
    if hasattr(model, "device"):
        device = model.device

    print(f"Loading checkpoint weights from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    query_lut = checkpoint["query_lut"].to(device)

    print(f"Loading Dense model '{args.dense_model}' on {device}...")
    dense_model = AutoModel.from_pretrained(args.dense_model).to(device)
    dense_tokenizer = AutoTokenizer.from_pretrained(args.dense_model)
    dense_model.eval()
    model.eval()

    # Load targets
    with open(args.targets_path, "r") as f:
        all_targets = json.load(f)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    vocab_size = model.config.vocab_size

    for split in splits:
        print(f"\n=======================================================")
        print(f"  EVALUATING SPLIT: {split.upper()}")
        print(f"=======================================================")
        
        split_entries = [t for t in all_targets if t.get("split") == split]
        if not split_entries:
            print(f"No targets found for split '{split}' in {args.targets_path}")
            continue

        print(f"Found {len(split_entries)} test queries for {split}.")

        # Collect all candidate documents across queries in this split
        needed_doc_ids = set()
        for e in split_entries:
            needed_doc_ids.update(e["candidate_ids"])
        print(f"Total unique candidate documents to encode: {len(needed_doc_ids)}")

        # Load docs dataset from HuggingFace
        print(f"Loading documents dataset for split '{split}'...")
        docs_ds = load_dataset("mm-bright/MM-BRIGHT", "documents", split=split)
        doc_dict = {}
        for row in tqdm(docs_ds, desc="Scanning docs_ds"):
            d_id = row["id"]
            if d_id in needed_doc_ids:
                doc_text = row.get("content", row.get("text", row.get("caption", "")))
                doc_dict[d_id] = doc_text if (doc_text and doc_text.strip()) else "[No text]"
                if len(doc_dict) == len(needed_doc_ids):
                    break

        print(f"Loaded {len(doc_dict)} / {len(needed_doc_ids)} candidate texts.")

        # Pre-encode all candidate documents (V-SPLADE and Dense)
        cand_list_ids = list(doc_dict.keys())
        cand_texts = [doc_dict[d_id] for d_id in cand_list_ids]

        print("Encoding candidate documents with V-SPLADE student...")
        w_p_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(cand_texts), args.batch_size), desc="V-SPLADE Doc Batches"):
                sub_texts = cand_texts[i : i + args.batch_size]
                doc_inputs = tokenizer(
                    sub_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                outputs = model.model(
                    input_ids=doc_inputs["input_ids"],
                    attention_mask=doc_inputs["attention_mask"],
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
                attention_mask = doc_inputs["attention_mask"].to(hidden_states.device).unsqueeze(-1)

                chunk_size = 20000
                z_p_parts = []
                for start in range(0, vocab_size, chunk_size):
                    weight_chunk = model.lm_head.weight[start : start + chunk_size].to(hidden_states.device)
                    logits_chunk = torch.matmul(hidden_states, weight_chunk.t())
                    logits_chunk = logits_chunk * attention_mask + (1 - attention_mask) * -1e9
                    z_p_chunk, _ = torch.max(logits_chunk, dim=1)
                    z_p_parts.append(z_p_chunk)

                z_p = torch.cat(z_p_parts, dim=1)
                w_p = torch.log1p(torch.relu(z_p))
                w_p_list.append(w_p.cpu())

        w_p_all = torch.cat(w_p_list, dim=0) # (num_candidates, vocab_size)

        print("Encoding candidate documents with Dense model (BGE)...")
        dense_embeds_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(cand_texts), args.dense_batch_size), desc="Dense Doc Batches"):
                sub_texts = cand_texts[i : i + args.dense_batch_size]
                inputs = dense_tokenizer(
                    sub_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                outputs = dense_model(**inputs)
                token_embeddings = outputs[0]
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                doc_features = sum_embeddings / sum_mask
                doc_features = F.normalize(doc_features, p=2, dim=1)
                dense_embeds_list.append(doc_features.cpu())

        dense_embeds_all = torch.cat(dense_embeds_list, dim=0).to(device) # (num_candidates, dim)
        id2idx = {d_id: idx for idx, d_id in enumerate(cand_list_ids)}

        # Evaluate across the 3 sanitization strategies
        sanitizer_results = {}
        for s_name, s_func in SANITIZERS.items():
            mrr_list = []
            ndcg_list = []

            for entry in split_entries:
                q_text = entry["query_text"]
                gold_ids = entry["ground_truth_ids"]
                cand_ids = entry["candidate_ids"]

                cand_indices = [id2idx[d_id] for d_id in cand_ids if d_id in id2idx]
                if not cand_indices:
                    continue

                # 1. Clean query
                q_cleaned = s_func(q_text, domain=split)
                q_tok_ids = tokenizer(q_cleaned, add_special_tokens=False)["input_ids"]
                if not q_tok_ids:
                    q_tok_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

                # 2. Sparse Student scores on candidate pool
                w_q = torch.zeros(vocab_size, dtype=torch.float32)
                with torch.no_grad():
                    w_q_weights = F.softplus(query_lut[q_tok_ids]).cpu()
                    w_q[q_tok_ids] = w_q_weights

                w_p_cands = w_p_all[cand_indices] # shape: (pool_size, vocab_size)
                scores_sparse = torch.sum(w_q.unsqueeze(0) * w_p_cands, dim=1)

                if args.use_prf and len(scores_sparse) >= args.prf_k:
                    top_k_inds = torch.topk(scores_sparse, k=args.prf_k).indices
                    feedback_docs = w_p_cands[top_k_inds]
                    w_q_feedback = torch.mean(feedback_docs, dim=0)
                    w_q_exp = (1.0 - args.prf_alpha) * w_q + args.prf_alpha * w_q_feedback
                    top_vals, top_inds = torch.topk(w_q_exp, k=20)
                    w_q_pruned = torch.zeros_like(w_q_exp)
                    w_q_pruned[top_inds] = top_vals
                    scores_sparse = torch.sum(w_q_pruned.unsqueeze(0) * w_p_cands, dim=1)

                scores_sparse = scores_sparse.to(device)

                # 3. Dense scores on candidate pool
                with torch.no_grad():
                    q_inputs = dense_tokenizer([q_text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    q_outputs = dense_model(**q_inputs)
                    q_tok_emb = q_outputs[0]
                    q_att = q_inputs["attention_mask"].unsqueeze(-1).expand(q_tok_emb.size()).float()
                    q_feat = torch.sum(q_tok_emb * q_att, 1) / torch.clamp(q_att.sum(1), min=1e-9)
                    q_feat = F.normalize(q_feat, p=2, dim=1)

                    dense_cands = dense_embeds_all[cand_indices]
                    scores_dense = torch.sum(q_feat * dense_cands, dim=1)

                # 4. Hybrid score fusion
                norm_sparse = min_max_normalize(scores_sparse)
                norm_dense = min_max_normalize(scores_dense)
                scores_hybrid = (1.0 - args.beta) * norm_sparse + args.beta * norm_dense

                # Rank candidates
                ranked_pool_indices = torch.argsort(scores_hybrid, descending=True).tolist()
                ranked_doc_ids = [cand_ids[idx] for idx in ranked_pool_indices]

                # MRR@10
                mrr_val = 0.0
                for r_idx, d_id in enumerate(ranked_doc_ids[:10]):
                    if d_id in gold_ids:
                        mrr_val = 1.0 / (r_idx + 1)
                        break
                mrr_list.append(mrr_val)

                # NDCG@10
                dcg = 0.0
                for r_idx, d_id in enumerate(ranked_doc_ids[:10]):
                    if d_id in gold_ids:
                        dcg += 1.0 / (np.log2(r_idx + 2))
                idcg = 0.0
                for r_idx in range(min(10, len(gold_ids))):
                    idcg += 1.0 / (np.log2(r_idx + 2))
                ndcg_val = dcg / idcg if idcg > 0.0 else 0.0
                ndcg_list.append(ndcg_val)

            avg_mrr = np.mean(mrr_list)
            avg_ndcg = np.mean(ndcg_list)
            sanitizer_results[s_name] = {"MRR@10": avg_mrr, "NDCG@10": avg_ndcg}

        print(f"\n--- COMPARATIVE RESULTS FOR {split.upper()} ---")
        print(f"{'Sanitizer Mode':<15} | {'MRR@10':<10} | {'NDCG@10':<10} | {'Delta vs Current':<18}")
        print("-" * 65)
        base_ndcg = sanitizer_results["current"]["NDCG@10"]
        base_mrr = sanitizer_results["current"]["MRR@10"]
        for s_name in ["current", "raw", "smart"]:
            res = sanitizer_results[s_name]
            mrr = res["MRR@10"]
            ndcg = res["NDCG@10"]
            delta = f"{(ndcg - base_ndcg):+.4f} ({(ndcg - base_ndcg)/base_ndcg*100:+.1f}%)" if base_ndcg > 0 else "0.0%"
            print(f"{s_name:<15} | {mrr:.4f}     | {ndcg:.4f}     | {delta}")

    print("\nSanity Test Completed Successfully.")

if __name__ == "__main__":
    main()
