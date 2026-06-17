import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    # Detect available result files
    synthetic_path = "lqe_results.json"
    real_path = "lqe_real_results.json"
    nfcorpus_path = "lqe_nfcorpus_results.json"
    
    synthetic_exists = os.path.exists(synthetic_path)
    real_exists = os.path.exists(real_path)
    nfcorpus_exists = os.path.exists(nfcorpus_path)
    
    active_rows = []
    if synthetic_exists:
        active_rows.append("synthetic")
    if real_exists:
        active_rows.append("real")
    if nfcorpus_exists:
        active_rows.append("nfcorpus")
        
    if not active_rows:
        print("Error: No benchmark result files found. Run the evaluation scripts first:")
        print("  - lqe_evaluation.py (Synthetic Memory)")
        print("  - real_dataset_eval.py (LongMemEval)")
        print("  - nfcorpus_eval.py (NFCorpus)")
        return

    methods = ["grep", "vector", "lqe_grep", "lqe_grep_v2", "cursor_hybrid", "sandboxed_python", "iterative_grep"]
    method_labels = {
        "grep": "Vanilla Grep",
        "vector": "Vector Search",
        "lqe_grep": "LQE-Grep (v1)",
        "lqe_grep_v2": "LQE-Grep v2 (Ours)",
        "cursor_hybrid": "Cursor Hybrid (Sim)",
        "sandboxed_python": "ChatGPT Python (Sim)",
        "iterative_grep": "Claude Code Grep (Sim)"
    }
    method_colors = {
        "grep": "#7f7f7f",            # Neutral Gray
        "vector": "#d62728",          # Warning Red/Coral
        "lqe_grep": "#9467bd",        # Muted Purple
        "lqe_grep_v2": "#1f77b4",     # Strong Blue
        "cursor_hybrid": "#bcbd22",   # Olive Green
        "sandboxed_python": "#ff7f0e", # Orange
        "iterative_grep": "#2ca02c"   # Green
    }
    method_markers = {
        "grep": "o",
        "vector": "s",
        "lqe_grep": "d",
        "lqe_grep_v2": "^",
        "cursor_hybrid": "v",
        "sandboxed_python": "*",
        "iterative_grep": "x"
    }

    # Set up styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 11
    
    n_rows = len(active_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows), squeeze=False)
    
    for row_idx, benchmark_type in enumerate(active_rows):
        ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1]
        
        if benchmark_type == "synthetic":
            # 1. Synthetic Memory Dialogue Sweep
            with open(synthetic_path, "r") as f:
                data = json.load(f)
                
            noise_levels = sorted([int(k) for k in data.keys()])
            
            # Check which methods are actually present in the data
            present_methods = [
                m for m in methods 
                if any(m in data[str(noise)] for noise in noise_levels)
            ]
            
            accuracies = {m: [] for m in present_methods}
            tokens = {m: [] for m in present_methods}
            
            for noise in noise_levels:
                noise_str = str(noise)
                for m in present_methods:
                    m_data = data[noise_str].get(m, {"accuracy": 0.0, "avg_tokens": 0.0})
                    accuracies[m].append(m_data.get("accuracy", 0.0) * 100)
                    tokens[m].append(m_data.get("avg_tokens", 0.0))
                    
            # Left panel: Accuracy
            for m in present_methods:
                ax1.plot(
                    noise_levels, 
                    accuracies[m], 
                    label=method_labels[m], 
                    color=method_colors[m], 
                    marker=method_markers[m], 
                    linewidth=2, 
                    markersize=8
                )
            ax1.set_xlabel("Noise Level (Number of Distractor Turns)", labelpad=6)
            ax1.set_ylabel("QA Accuracy (%)")
            ax1.set_title("Synthetic Memory: QA Accuracy vs. Noise")
            ax1.set_xticks(noise_levels)
            ax1.set_ylim(-5, 105)
            ax1.grid(True, linestyle="--", alpha=0.6)
            
            # Right panel: Tokens
            for m in present_methods:
                ax2.plot(
                    noise_levels, 
                    tokens[m], 
                    label=method_labels[m], 
                    color=method_colors[m], 
                    marker=method_markers[m], 
                    linewidth=2, 
                    markersize=8
                )
            ax2.set_xlabel("Noise Level (Number of Distractor Turns)", labelpad=6)
            ax2.set_ylabel("Avg Context Footprint (Tokens)")
            ax2.set_title("Synthetic Memory: Token Footprint vs. Noise")
            ax2.set_xticks(noise_levels)
            ax2.grid(True, linestyle="--", alpha=0.6)
            
        elif benchmark_type == "real":
            # 2. LongMemEval (Real Dataset)
            with open(real_path, "r") as f:
                data = json.load(f)
                
            present_methods = [m for m in methods if m in data]
            
            acc_vals = [data[m]["accuracy"] * 100 for m in present_methods]
            tok_vals = [data[m]["avg_tokens"] for m in present_methods]
            
            bar_labels = [method_labels[m] for m in present_methods]
            bar_colors = [method_colors[m] for m in present_methods]
            
            # Left panel: Accuracy
            bars1 = ax1.bar(bar_labels, acc_vals, color=bar_colors, edgecolor="black", alpha=0.85)
            # Add values on top of bars
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0, 
                    yval + 1, 
                    f"{yval:.1f}%", 
                    ha='center', 
                    va='bottom', 
                    fontsize=9,
                    fontweight="bold"
                )
            ax1.set_ylabel("QA Accuracy (%)")
            ax1.set_title("LongMemEval: QA Accuracy")
            ax1.set_ylim(0, 110)
            ax1.grid(True, linestyle="--", alpha=0.4, axis='y')
            ax1.tick_params(axis='x', rotation=20)
            
            # Right panel: Tokens
            bars2 = ax2.bar(bar_labels, tok_vals, color=bar_colors, edgecolor="black", alpha=0.85)
            # Add values on top of bars
            max_tok = max(tok_vals) if tok_vals else 1.0
            for bar in bars2:
                yval = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0, 
                    yval + (max_tok * 0.01), 
                    f"{yval:.0f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=9,
                    fontweight="bold"
                )
            ax2.set_ylabel("Avg Context Footprint (Tokens)")
            ax2.set_title("LongMemEval: Token Footprint")
            ax2.grid(True, linestyle="--", alpha=0.4, axis='y')
            ax2.tick_params(axis='x', rotation=20)
            
        elif benchmark_type == "nfcorpus":
            # 3. BEIR NFCorpus
            with open(nfcorpus_path, "r") as f:
                data = json.load(f)
                
            present_methods = [m for m in methods if m in data]
            
            acc_vals = [data[m]["success_at_3"] * 100 for m in present_methods]
            tok_vals = [data[m]["avg_tokens"] for m in present_methods]
            
            bar_labels = [method_labels[m] for m in present_methods]
            bar_colors = [method_colors[m] for m in present_methods]
            
            # Left panel: Success@3
            bars1 = ax1.bar(bar_labels, acc_vals, color=bar_colors, edgecolor="black", alpha=0.85)
            # Add values on top of bars
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0, 
                    yval + 1, 
                    f"{yval:.1f}%", 
                    ha='center', 
                    va='bottom', 
                    fontsize=9,
                    fontweight="bold"
                )
            ax1.set_ylabel("Success@3 (%)")
            ax1.set_title("NFCorpus: Success@3 Retrieval Rate")
            ax1.set_ylim(0, 110)
            ax1.grid(True, linestyle="--", alpha=0.4, axis='y')
            ax1.tick_params(axis='x', rotation=20)
            
            # Right panel: Tokens
            bars2 = ax2.bar(bar_labels, tok_vals, color=bar_colors, edgecolor="black", alpha=0.85)
            # Add values on top of bars
            max_tok = max(tok_vals) if tok_vals else 1.0
            for bar in bars2:
                yval = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0, 
                    yval + (max_tok * 0.01), 
                    f"{yval:.0f}", 
                    ha='center', 
                    va='bottom', 
                    fontsize=9,
                    fontweight="bold"
                )
            ax2.set_ylabel("Avg Context Footprint (Tokens)")
            ax2.set_title("NFCorpus: Token Footprint")
            ax2.grid(True, linestyle="--", alpha=0.4, axis='y')
            ax2.tick_params(axis='x', rotation=20)

    # Global title
    fig.suptitle("LQE-Grep Evaluation Benchmark Suite", fontsize=16, fontweight="bold", y=0.985)

    # Dynamic bottom margin adjust for legend
    if n_rows == 1:
        bottom_val = 0.22
        top_val = 0.86
    elif n_rows == 2:
        bottom_val = 0.12
        top_val = 0.92
    else:
        bottom_val = 0.08
        top_val = 0.94
        
    # Create manual legend handles for all methods using patches
    legend_handles = [
        mpatches.Patch(color=method_colors[m], label=method_labels[m])
        for m in methods
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=top_val, bottom=bottom_val)
    
    output_path = "evaluation_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    main()
