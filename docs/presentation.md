# Slide 1: Title Slide
# Bridging the Lexical-Semantic Gap in Agentic Search
## Lexical Query Expansion (LQE) via Harness-Level Regex Synthesis

**Esteban Schafir · FIU Research · June 2026**

---

# Slide 2: Background: "Is Grep All You Need?" (Sen et al., 2026)
### Analysis of the Baseline Study (arXiv:2605.15184v1)
*   **What it does**:
    *   Systematically compares **lexical search (`grep`)** vs. **semantic search (`vector`)** in agent workflows.
    *   Tests interactive tool-calling loops across a custom harness (`Chronos`) and provider CLI agents (`Claude Code`, `Codex`, `Gemini CLI`).
    *   Compares context presentation styles: **inline delivery** vs. **programmatic delivery** (writing results to a file for agents to read).
    *   Sweeps noise levels by progressively mixing in unrelated distractor conversation sessions (5 to 30+).
*   **Key Findings**:
    *   Inline `grep` consistently beats dense vector search in accuracy when exact keywords/events are present.
    *   Harness orchestration mechanics (prompts, formatting, tool limits) impact performance as much as retriever choice itself.

---
# Slide 3: What the Paper Lacks (The Retrieval Gaps)
### The Lexical-Semantic Dilemma
1.  **The Vocabulary Mismatch Trap**:
    *   While `grep` is precise, it is completely brittle to word variation.
    *   If the query asks for `"vehicle"` but the document uses `"sedan"`, `grep` returns **0 matches** (0.0% accuracy in our tests).
2.  **Vector Search Noise (Action Template Overlap)**:
    *   Dense retrieval captures meaning but fails under noise. It retrieves irrelevant items (e.g., `"bought a white trenchcoat"`) instead of target entities because the embedding space prioritizes action structures (`"bought"`, `"buy"`) over entity classes.
3.  **Lack of Automated Synthesis Middleware**:
    *   The paper assumes agents must manually iterate on queries or rely on pre-extracted schemas. It lacks an automated, dynamic translation layer to bridge the lexical-semantic gap.

---

# Slide 4: Why We Propose LQE (Lexical Query Expansion)
### Harness-Level Regex Query Synthesis
We propose LQE to resolve the core limitations of both `grep` and `vector` search:
*   **Semantic Recall + Lexical Precision**: A lightweight LLM middleware step inside the harness expands semantic concepts into regex alternation groups (e.g., `vehicle` $\to$ `(sedan|coupe|suv|...)`).
*   **Semantic Entity Locking**: Regex acts as a strict lexical filter, blocking vector search distractors (like apparel buying actions) by matching only specified category members.
*   **Cognitive Offloading**: Offloads query planning from the main agent to the harness middleware, avoiding agent collapse on smaller backbones.
*   **Zero-Index Infrastructure**: Reaps the benefits of semantic search directly on raw text files with zero embedding or vector database overhead.

---

# Slide 5: Proposed Solution: Lexical Query Expansion (LQE)
### Harness-Level Regex Query Synthesis
Instead of forcing the LLM agent to think of synonyms or running expensive vector indexes, we implement **automatic LQE in the harness middleware**:

```
Agent issues Search -> [Harness LQE Prompt] -> LLM Synthesizes regex -> Grep search on raw files
```

*   **Zero-Index Semantic Search**: Achieves semantic-like recall on raw text files with zero index build/maintenance overhead.
*   **Cognitive Offloading**: Bypasses the agent's query planning phase, reducing failures on mid-sized models.

---

# Slide 6: How the Benchmark Dataset is Created
### Programmatic Dialogue Generation (Burying a Needle in a Haystack)
1. **Define Categories & Templates**:
   * *Category*: `vehicle` (Target item: `sedan`)
   * *Oracle Template*: `"The user purchased a {color} {item} yesterday."`
   * *Synonym Query*: `"What color vehicle did the user buy?"` (forces a vocabulary mismatch: `vehicle` vs `sedan`)
2. **Generate the Oracle ("The Needle")**:
   * Pick random color & specific item: E.g., `emerald` and `sedan`.
   * *Oracle turn*: `"[User]: The user purchased a emerald sedan yesterday."`
3. **Mix in Distractors ("The Haystack")**:
   * *General chat*: `"The weather was quite nice today, perfect for walking."`
   * *Same-category distractor*: `"The user was looking at a blue motorcycle online."`
   * *Action-overlap distractor*: `"The user bought a matching white trenchcoat for the party."` (uses verb `bought`)
4. **Shuffle, Format, & Noise Sweeps**:
   * Distribute turns randomly, prefixing with `[User]` or `[Assistant]`.
   * Sweep the noise levels (10, 20, and 35 distractor turns) to analyze degradation.

---

# Slide 7: Pilot Results: LQE-Grep vs. Vector vs. Grep
### Accuracy & Context Footprint at 35 Distractors (Qwen2.5-1.5B)
*   **Vanilla Grep**: **0.0% Accuracy** (Average Context Footprint: 8.0 tokens).
    *   *Result*: 0 retrieved lines due to vocabulary mismatch.
*   **Vector Search**: **0.0% Accuracy** (Average Context Footprint: 42.2 tokens).
    *   *Result*: Retrieved irrelevant turns due to action template similarities.
*   **LQE-Grep (Ours)**: **100.0% Accuracy** (Average Context Footprint: 63.7 tokens).
    *   *Result*: Retrieved the target `"emerald sedan"` turn exactly.

---

# Slide 8: Real-World Benchmark: LongMemEval Evaluation
### Results on Real User-Assistant Conversations (~100k tokens per query)
*   **The Setup**: Evaluated on 10 random `single-session-user` queries from `longmemeval_s_cleaned.json`.
*   **Summary Table**:
    *   **Vanilla Grep**: **70.0% Accuracy** (Avg context: 717.1 tokens)
    *   **LQE-Grep (v1)**: **60.0% Accuracy** (Avg context: 3,933.1 tokens)
    *   **LQE-Grep v2 (Ours)**: **60.0% Accuracy** (Avg context: **2,897.7 tokens** - **26.3% reduction**)
    *   **Vector Search**: **0.0% Accuracy** (Avg context: 53.2 tokens)
*   **The Truncation Bottleneck**:
    *   Vector search failed completely because embedding 500+ turns was too slow, forcing truncation to the first 150 turns. The answer (in session 51) was missed entirely.
    *   Lexical search scanned all 500+ turns in milliseconds, capturing the needle. LQE v1 suffered from matching common expanded words (e.g. "name", "shop"), which LQE v2 successfully prunes via dynamic local turn-frequency filtering.

---

# Slide 9: Real-World Medical Benchmark: BEIR NFCorpus Evaluation
### Zero-Shot Document Retrieval (Success@3) on a Haystack of 50 Abstracts
*   **The Setup**: Evaluated on all 323 test queries comparing patient-written terms to medical documents.
*   **Summary Table**:
    *   **Vanilla Grep**: 50.42% Success@3 (Avg context: 1,949.6 tokens)
    *   **LQE-Grep (v1)**: 56.30% Success@3 (Avg context: 5,689.8 tokens)
    *   **LQE-Grep v2 (Ours)**: **57.98% Success@3** (Avg context: **3,182.3 tokens** - **44.1% reduction**)
    *   **Vector Search**: 15.13% Success@3
*   **The Precision Gap**:
    *   Vector search failed (15.13% success) because embeddings prioritize general semantic associations (e.g., retrieving generic blood cell papers for *"Dragon's Blood"*).
    *   Lexical search restricts matching to actual entity names, maintaining high precision. LQE v2 outperforms LQE v1 in recall while pruning domain-specific high-frequency terminology via global corpus-level Document Frequency (DF) filtering.

---

# Slide 10: Insights: Semantic Entity Locking
### Why Vector Search Failed
*   Vector search was confused by **action template overlap**. When the query was *"What color vehicle did the user buy?"*, it retrieved:
    `[User]: The user bought a matching white trenchcoat for the party.`
*   Because `"bought"` and `"trenchcoat"` share semantic overlaps with buying actions, vector search prioritizes structure over the target entity category.

### Why LQE-Grep Succeeded
*   The LLM expanded `"vehicle"` to `(sedan|coupe|motorcycle|suv|truck|...)`.
*   The regex acts as a strict **lexical filter**, forcing the tool to match only matching entity classes and ignoring action overlap.

---

# Slide 11: Critical Optimization: Word Boundaries
### Substring Collision Bug
*   In early runs, the query expansion for `vehicle` included the word `car`.
*   This caused false positive matches on distractors containing `car` as a substring (e.g. `cardigan`, `cardiologist`).
*   Accuracy at 35 distractors dropped to **16.67%**.

### The Fix
*   Wrapped LQE regex strings in **word boundaries** (`\b`):
    `\b(sedan|coupe|...|car|...)\b`
*   Eliminated all false positives, immediately restoring accuracy to **100.0%**.

---

# Slide 12: Scaled Plan & Next Steps
1.  **Stop-Word & Common-Word Filtering**:
    *   Optimize LQE query expansion to prune overly broad terms (like `name` or `shop`) that cause high token recall in large corpora.
2.  **Scale Real-World Evaluation**:
    *   Evaluate on the full 500-question LongMemEval dataset and the complete 3,000+ document BEIR NFCorpus benchmark.
3.  **Draft Paper Layout**:
    *   Present LQE-Grep as a compute-efficient, zero-index alternative to RAG for dynamic environments.
    *   Detail how the compute complexity of embedding-based search forces truncation in long-context, failing where lexical search succeeds.
