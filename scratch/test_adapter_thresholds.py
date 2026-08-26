import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import collections

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)

class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))
    def forward(self, x):
        return self.linear(x)

# Load bottleneck adapter checkpoint
checkpoint = torch.load('results/vsplade_student_checkpoint.pt', map_location=device)
hidden_dim = model.config.hidden_size
model_dtype = next(model.parameters()).dtype
adapter = BottleneckAdapter(hidden_dim).to(device).to(model_dtype)
adapter.load_state_dict(checkpoint["adapter_state_dict"])
adapter.eval()

query_lut = checkpoint["query_lut"].to(device)

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

def clean_query_text(text):
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

print("Loading dataset...")
docs_ds = load_dataset('mm-bright/MM-BRIGHT', "documents", split='academia')
queries_ds = load_dataset('mm-bright/MM-BRIGHT', "examples", split='academia')

queries_ds = list(queries_ds)[:20]

gold_doc_ids = set()
for q in queries_ds:
    gold_doc_ids.update(q['gold_ids'])
    
test_docs = []
for doc in docs_ds:
    if doc['id'] in gold_doc_ids or len(test_docs) < 5000:
        test_docs.append(doc)

print(f"Testing on {len(test_docs)} docs and {len(queries_ds)} queries.")
corpus_ids = [d['id'] for d in test_docs]
vocab_size = model.config.vocab_size

# Tokenize docs
doc_freqs = collections.Counter()
for d in test_docs:
    unique_tokens = set(tokenizer(d['content'], add_special_tokens=False)["input_ids"])
    doc_freqs.update(unique_tokens)

num_docs = len(test_docs)
idf = {}
for token_id, df in doc_freqs.items():
    idf[token_id] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

print("Encoding docs...")
w_p_list = []
with torch.no_grad():
    for start_idx in range(0, len(test_docs), 64):
        batch = test_docs[start_idx : start_idx + 64]
        doc_texts = [d['content'] for d in batch]
        
        doc_inputs = tokenizer(doc_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        outputs = model.model(**doc_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        projected = adapter(hidden_states)
        
        attention_mask = doc_inputs["attention_mask"].unsqueeze(-1)
        z_p_parts = []
        for start in range(0, vocab_size, 20000):
            weight_chunk = model.lm_head.weight[start : start + 20000]
            logits_chunk = torch.matmul(projected, weight_chunk.t())
            logits_chunk = logits_chunk * attention_mask + (1 - attention_mask) * -1e9
            z_p_chunk, _ = torch.max(logits_chunk, dim=1)
            z_p_parts.append(z_p_chunk)
            
        z_p = torch.cat(z_p_parts, dim=1)
        w_p_list.append(z_p.cpu())

z_p_all = torch.cat(w_p_list, dim=0)

# Evaluate with different thresholds
for threshold in [0.0, 4.0, 8.0, 12.0]:
    mrr_total = 0.0
    count = 0
    
    # Pre-apply threshold to w_p
    w_p_all = torch.log1p(torch.relu(z_p_all - threshold))
    
    for q in queries_ds:
        q_text = q['query']
        gold_ids = q['gold_ids']
        
        stopped_text = clean_query_text(q_text)
        query_token_ids = tokenizer(stopped_text, add_special_tokens=False)["input_ids"]
        if not query_token_ids:
             continue
             
        w_q = torch.zeros(vocab_size, dtype=torch.float32)
        with torch.no_grad():
            w_q_weights = F.softplus(query_lut[query_token_ids]).cpu()
            
            # Apply IDF scaling
            for i, token_id in enumerate(query_token_ids):
                token_idf = idf.get(token_id, math.log(1.0 + num_docs))
                w_q_weights[i] = w_q_weights[i] * token_idf
                
            w_q[query_token_ids] = w_q_weights
            
        scores = torch.sum(w_q.unsqueeze(0) * w_p_all, dim=1)
        ranked_indices = torch.argsort(scores, descending=True).tolist()
        ranked_doc_ids = [corpus_ids[idx] for idx in ranked_indices]
        
        mrr_val = 0.0
        for rank_idx, doc_id in enumerate(ranked_doc_ids[:10]):
            if doc_id in gold_ids:
                mrr_val = 1.0 / (rank_idx + 1)
                break
        mrr_total += mrr_val
        count += 1
        
    print(f"Logit Threshold: {threshold:<5} | Average MRR@10: {mrr_total / count:.4f}")
