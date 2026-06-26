import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")
    results_json_path = os.path.join(results_dir, "mugi_comparison_results.json")
    
    if not os.path.exists(results_json_path):
        print(f"Error: {results_json_path} does not exist.")
        return
        
    with open(results_json_path, "r") as f:
        data = json.load(f)
        
    # Order methods for logical comparison
    methods = [
        "grep",
        "lqe_grep",
        "lqe_grep_v2",
        "lqe_grep_v3_stemmed",
        "lqe_grep_v3_weighted",
        "lqe_grep_v4",
        "bm25",
        "mugi_gpt35",
        "mugi_qwen",
        "mugi_gpt4"
    ]
    
    labels = {
        "grep": "Vanilla Grep",
        "lqe_grep": "LQE-Grep v1",
        "lqe_grep_v2": "LQE-Grep v2 (Pruned)",
        "lqe_grep_v3_stemmed": "LQE-Grep v3 (Stemmed)",
        "lqe_grep_v3_weighted": "LQE-Grep v3 (Weighted)",
        "lqe_grep_v4": "LQE-Grep v4 (Stem+IDF)",
        "bm25": "BM25 Baseline",
        "mugi_gpt35": "MuGI (GPT-3.5)",
        "mugi_qwen": "MuGI (Qwen-1.5B)",
        "mugi_gpt4": "MuGI (GPT-4)"
    }
    
    # Extract metrics
    accuracies = [data[m]["success_at_3"] * 100 for m in methods]
    context_tokens = [data[m]["avg_tokens"] for m in methods]
    generation_tokens = [data[m]["avg_search_tokens"] for m in methods]
    
    colors = [
        "#7f7f7f", # Vanilla Grep (Gray)
        "#bcbd22", # LQE-Grep v1 (Olive)
        "#17becf", # LQE-Grep v2 (Light Blue)
        "#9467bd", # LQE-Grep v3 Stemmed (Muted Purple)
        "#ff7f0e", # LQE-Grep v3 Weighted (Orange)
        "#1f77b4", # LQE-Grep v4 (Strong Blue)
        "#2ca02c", # BM25 (Green)
        "#e377c2", # MuGI GPT-3.5 (Pink)
        "#8c564b", # MuGI Qwen (Brown)
        "#d62728"  # MuGI GPT-4 (Red)
    ]
    
    # Set up matplotlib style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Left Panel: Success@3 Accuracy (%)
    bar_labels = [labels[m] for m in methods]
    bars1 = ax1.barh(bar_labels, accuracies, color=colors, edgecolor="black", alpha=0.8)
    for bar in bars1:
        width = bar.get_width()
        ax1.text(
            width + 1.0, 
            bar.get_y() + bar.get_height() / 2.0, 
            f"{width:.1f}%", 
            ha="left", 
            va="center", 
            fontsize=9, 
            fontweight="bold"
        )
    ax1.set_xlabel("Success@3 Accuracy (%)", fontweight="bold")
    ax1.set_title("Retrieval Accuracy on NFCorpus", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlim(0, 90)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="x")
    
    # 2. Right Panel: Computational Footprint Trade-off
    x = np.arange(len(methods))
    width = 0.35
    
    bars_ctx = ax2.bar(x - width/2, context_tokens, width, label="Avg Context Footprint", color="#1f77b4", edgecolor="black", alpha=0.7)
    bars_gen = ax2.bar(x + width/2, generation_tokens, width, label="Avg Generation Cost", color="#ff7f0e", edgecolor="black", alpha=0.7)
    
    ax2.set_ylabel("Tokens", fontweight="bold")
    ax2.set_title("Token Footprint: Context vs Generation Cost", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    
    # Add values on top of footprint bars
    for bar in bars_ctx:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 100,
                f"{yval:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )
            
    # Add values on top of generation bars
    for bar in bars_gen:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 100,
                f"{yval:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90
            )
            
    ax2.set_ylim(0, max(max(context_tokens), max(generation_tokens)) + 1000)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, "mugi_comparison_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    main()
