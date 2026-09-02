import re
from datasets import load_dataset
from transformers import AutoTokenizer

stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'could', 'didn', 'doesn', 'hadn',
    'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])

def clean_query_current(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

def clean_query_smart(text, domain=None):
    text = re.sub(r'<[^>]+>', '', text)
    words = text.split()
    kept_words = []
    for w in words:
        core = re.sub(r'^\W+|\W+$', '', w).lower()
        if core in stop_words:
            if w.lower() == core:
                continue
        kept_words.append(w)
    return ' '.join(kept_words)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

for split in ['math', 'askubuntu']:
    ds = load_dataset('mm-bright/MM-BRIGHT', 'examples', split=split)
    print(f"\n==================== SPLIT: {split} ====================")
    for i in range(min(3, len(ds))):
        raw = ds[i]['query']
        curr = clean_query_current(raw)
        smart = clean_query_smart(raw, domain=split)
        
        curr_toks = tokenizer.tokenize(curr)[:12]
        smart_toks = tokenizer.tokenize(smart)[:12]
        
        print(f"\n--- Example {i+1} ---")
        print(f"RAW:        {repr(raw[:80])}...")
        print(f"CURRENT:    {repr(curr[:80])}...")
        print(f"SMART:      {repr(smart[:80])}...")
        print(f"Tokens (Current): {curr_toks}")
        print(f"Tokens (Smart):   {smart_toks}")
