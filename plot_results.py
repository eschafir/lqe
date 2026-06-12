import json
import matplotlib.pyplot as plt

def main():
    # Load results from json
    with open("lqe_results.json", "r") as f:
        data = json.load(f)
        
    noise_levels = sorted([int(k) for k in data.keys()])
    
    methods = ["grep", "vector", "lqe_grep"]
    method_labels = {
        "grep": "Vanilla Grep",
        "vector": "Vector Search",
        "lqe_grep": "LQE-Grep (Ours)"
    }
    method_colors = {
        "grep": "#7f7f7f",      # Neutral Gray
        "vector": "#d62728",    # Warning Red/Coral
        "lqe_grep": "#1f77b4"   # Strong Blue
    }
    method_markers = {
        "grep": "o",
        "vector": "s",
        "lqe_grep": "^"
    }
    
    accuracies = {m: [] for m in methods}
    tokens = {m: [] for m in methods}
    
    for noise in noise_levels:
        noise_str = str(noise)
        for m in methods:
            accuracies[m].append(data[noise_str][m]["accuracy"] * 100)
            tokens[m].append(data[noise_str][m]["avg_tokens"])
            
    # Set up styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 11
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # 1. Accuracy vs Noise
    for m in methods:
        ax1.plot(
            noise_levels, 
            accuracies[m], 
            label=method_labels[m], 
            color=method_colors[m], 
            marker=method_markers[m], 
            linewidth=2, 
            markersize=8
        )
    ax1.set_xlabel("Noise Level (Number of Distractor Turns)", labelpad=10)
    ax1.set_ylabel("QA Accuracy (%)")
    ax1.set_title("QA Accuracy vs. Context Noise")
    ax1.set_xticks(noise_levels)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # 2. Token Footprint vs Noise
    for m in methods:
        ax2.plot(
            noise_levels, 
            tokens[m], 
            label=method_labels[m], 
            color=method_colors[m], 
            marker=method_markers[m], 
            linewidth=2, 
            markersize=8
        )
    ax2.set_xlabel("Noise Level (Number of Distractor Turns)", labelpad=10)
    ax2.set_ylabel("Avg Context Footprint (Tokens)")
    ax2.set_title("Token Footprint vs. Context Noise")
    ax2.set_xticks(noise_levels)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    # Place legend on top or bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.18)
    
    output_path = "evaluation_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    main()
