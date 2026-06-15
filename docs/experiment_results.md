# LQE-Grep Experiment Results & Walkthrough

**FIU Research · June 2026**

---

We have successfully set up the benchmark environment, resolved the model loading dependencies, and executed a pilot run of the **Lexical Query Expansion (LQE) vs. Grep vs. Vector Search** experiment.

Below is the technical log of the run and the findings from our analysis.

---

## 1. Core Results Table

The pilot run evaluated 6 query categories (Vehicles, Occupations, Residences, Pets, Refreshments, and Apparel) using the local `Qwen2.5-1.5B-Instruct` model across three distraction noise levels:

| Noise (Distractors) | Retrieval Method | QA Accuracy | Avg Context Footprint (Tokens) | Status |
|---|---|---|---|---|
| **10 turns** | Vanilla `grep` | 0.0% | 5.7 | ✗ Zero recall |
| | **LQE-Grep** | **83.3%** | **45.7** | **✓ Strong** |
| | Vector Search | 33.3% | 42.5 | ✗ Moderate |
| **20 turns** | Vanilla `grep` | 0.0% | 8.5 | ✗ Zero recall |
| | **LQE-Grep** | **83.3%** | **62.0** | **✓ Strong** |
| | Vector Search | 0.0% | 42.5 | ✗ Collapsed |
| **35 turns** | Vanilla `grep` | 0.0% | 8.0 | ✗ Zero recall |
| | **LQE-Grep** | **100.0%** | **63.7** | **✓ Perfect** |
| | Vector Search | 0.0% | 42.2 | ✗ Collapsed |

---

## 2. Deep Technical Insights

The trace logs from our runs reveal three highly distinct behaviors representing the core mechanics of each retrieval system:

### A. The Vocabulary Mismatch Problem (Vanilla Grep)
When the agent queries `"What color vehicle did the user buy?"`, vanilla `grep` searches for the word `vehicle` and matches **0 lines** because the conversation contains `"purchased a emerald sedan"`. As a result:
*   The context sent to the model is empty.
*   The model outputs `"Not applicable"` or `"None"`, leading to **0% accuracy** across all runs.

### B. Frame/Template Overlap Confusion (Vector Search)
Vector search computes embeddings from the model's internal hidden states. However, it fails catastrophically at higher noise levels (0% accuracy at 20 and 35 distractors). 
Looking at the logs, when querying `"What color vehicle did the user buy?"`, vector search retrieved:
1.  `[User]: The user bought a matching white trenchcoat for the party.` (Apparel)
2.  `[Assistant]: The user was looking at a brown motorcycle online.` (Vehicle distractor)
3.  `[Assistant]: The user was looking at a brown truck online.` (Vehicle distractor)

It **missed** the target line `"purchased a emerald sedan"`. 

**Why?** Because the target line used the word `"purchased"`, whereas the apparel line used the word `"bought"`, which overlaps semantically with the query phrase `"user buy"`. The vector embedding space is dominated by the **action template** (the act of buying something) rather than the **entity category** (vehicle vs apparel), leading to high-recall of incorrect categories.

### C. Semantic Entity Locking (LQE-Grep)
LQE-Grep uses the LLM to expand the query noun `vehicle` into a regex list of members of that category:
*   `Query`: "What color vehicle did the user buy?"
*   `Regex`: `\b(sedan|coupe|motorcycle|suv|truck|hatchback|convertible|automobile|vehicle|car|bike|scooter|van|wagon)\b`

Because of this lexical constraint, LQE-Grep is **completely immune** to frame-overlap confusion. It does not retrieve the apparel line because `"trenchcoat"` does not match any word in the regex. Instead, it successfully matched and retrieved `"purchased a emerald sedan"`, allowing the model to answer `"emerald"` with 100% correctness.

---

## 3. Critical Pilot Optimization: Word Boundaries

During our initial pilot runs, LQE-Grep scored 16.67% at high noise. Investigation of the logs revealed a critical regex boundary bug:
*   The generated pattern for `vehicle` included `car`.
*   The pattern matched the turn `"[Assistant]: The user got a new job working as a orange cardiologist."` and `"[Assistant]: The user bought a matching black cardigan..."` because `cardiologist` and `cardigan` contain the substring `car`.
*   This filled the context window with false positive distractors.

**The Fix:** We modified the LQE search code to wrap the LLM's generated regex pattern in word boundary tokens (`\b`). 

Once the pattern was constrained as `\b(sedan|coupe|...|car|...)\b`, accuracy immediately jumped from **16.67% to 100.0%** at 35 distractors.

---

## 4. Next Steps for Scale

With the pilot verifying the core hypothesis, you can now run a full experiment at scale:
1.  **Scale up the evaluation dataset**: Increase `--num-examples` from 1 to 10 (60 examples total) or 20 (120 examples total) to get statistically significant curves.
2.  **Generate plots**: We can write a script to plot the scaling behavior: **Accuracy vs. Noise (Distractors)** and **Context Token Footprint vs. Noise** for all three methods.
