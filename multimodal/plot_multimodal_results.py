import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")
    
    aro_clip_path = os.path.join(results_dir, "mlqe_aro_results.json")
    aro_vlm_path = os.path.join(results_dir, "vlm_aro_results.json")
    wino_clip_path = os.path.join(results_dir, "winoground_clip_results.json")
    wino_vlm_path = os.path.join(results_dir, "winoground_vlm_results.json")
    
    # 1. Load ARO Data
    aro_exists = os.path.exists(aro_clip_path) and os.path.exists(aro_vlm_path)
    aro_methods = []
    aro_scores = []
    
    if aro_exists:
        with open(aro_clip_path, "r") as f:
            aro_clip = json.load(f)
        with open(aro_vlm_path, "r") as f:
            aro_vlm = json.load(f)
            
        aro_methods = [
            "Vanilla CLIP",
            "M-LQE (Avg)",
            "M-LQE (Prod)",
            "M-LQE (Hybrid)",
            "M-LQE (Crop)",
            "M-LQE (Fusion)",
            f"VLM ({aro_vlm.get('model', 'Llama-3.2-Vision').split('/')[-1]})"
        ]
        
        aro_scores = [
            aro_clip.get("vanilla_accuracy", 0.0) * 100,
            aro_clip.get("mlqe_avg_accuracy", 0.0) * 100,
            aro_clip.get("mlqe_prod_accuracy", 0.0) * 100,
            aro_clip.get("mlqe_hybrid_accuracy", 0.0) * 100,
            aro_clip.get("mlqe_grounded_accuracy", 0.0) * 100,
            aro_clip.get("mlqe_fusion_accuracy", 0.0) * 100,
            aro_vlm.get("accuracy", 0.0) * 100
        ]
    
    # 2. Load Winoground Data
    wino_exists = os.path.exists(wino_clip_path) and os.path.exists(wino_vlm_path)
    wino_methods = []
    wino_text = []
    wino_image = []
    wino_group = []
    
    if wino_exists:
        with open(wino_clip_path, "r") as f:
            wino_clip = json.load(f)
        with open(wino_vlm_path, "r") as f:
            wino_vlm = json.load(f)
            
        wino_methods = [
            "Vanilla CLIP",
            "M-LQE (Avg)",
            "M-LQE (Prod)",
            "M-LQE (Hybrid)",
            f"VLM ({wino_vlm.get('model', 'Llama-3.2-Vision').split('/')[-1]})"
        ]
        
        clip_res = wino_clip.get("results", {})
        vlm_res = wino_vlm.get("results", {})
        
        wino_text = [
            clip_res.get("vanilla", {}).get("text_accuracy", 0.0) * 100,
            clip_res.get("mlqe_avg", {}).get("text_accuracy", 0.0) * 100,
            clip_res.get("mlqe_prod", {}).get("text_accuracy", 0.0) * 100,
            clip_res.get("mlqe_hybrid", {}).get("text_accuracy", 0.0) * 100,
            vlm_res.get("text_accuracy", 0.0) * 100
        ]
        
        wino_image = [
            clip_res.get("vanilla", {}).get("image_accuracy", 0.0) * 100,
            clip_res.get("mlqe_avg", {}).get("image_accuracy", 0.0) * 100,
            clip_res.get("mlqe_prod", {}).get("image_accuracy", 0.0) * 100,
            clip_res.get("mlqe_hybrid", {}).get("image_accuracy", 0.0) * 100,
            vlm_res.get("image_accuracy", 0.0) * 100
        ]
        
        wino_group = [
            clip_res.get("vanilla", {}).get("group_accuracy", 0.0) * 100,
            clip_res.get("mlqe_avg", {}).get("group_accuracy", 0.0) * 100,
            clip_res.get("mlqe_prod", {}).get("group_accuracy", 0.0) * 100,
            clip_res.get("mlqe_hybrid", {}).get("group_accuracy", 0.0) * 100,
            vlm_res.get("group_accuracy", 0.0) * 100
        ]
        
    if not aro_exists and not wino_exists:
        print("Error: No multimodal benchmark result files found inside results/ directory.")
        return
        
    # Styling setup
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    
    # 2 subplots layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color palette
    colors_aro = ["#7f7f7f", "#9467bd", "#e377c2", "#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]
    colors_wino = ["#3182bd", "#6baed6", "#9ecae1"] # Blues for text, image, group
    
    # Left subplot: ARO
    if aro_exists:
        bars = ax1.bar(aro_methods, aro_scores, color=colors_aro, edgecolor="black", alpha=0.85, width=0.6)
        ax1.set_ylabel("Accuracy (%)", fontweight="bold")
        ax1.set_title("ARO Visual Attribution: Attribute Binding", fontsize=12, fontweight="bold", pad=15)
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=30)
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 1.5,
                f"{yval:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight="bold"
            )
    else:
        ax1.text(0.5, 0.5, "ARO Data Missing", ha='center', va='center', fontsize=12)
        
    # Right subplot: Winoground
    if wino_exists:
        x = np.arange(len(wino_methods))
        width = 0.25
        
        rects1 = ax2.bar(x - width, wino_text, width, label="Text Score", color=colors_wino[0], edgecolor="black", alpha=0.85)
        rects2 = ax2.bar(x, wino_image, width, label="Image Score", color=colors_wino[1], edgecolor="black", alpha=0.85)
        rects3 = ax2.bar(x + width, wino_group, width, label="Group Score", color=colors_wino[2], edgecolor="black", alpha=0.85)
        
        ax2.set_ylabel("Accuracy (%)", fontweight="bold")
        ax2.set_title("Winoground: Compositional Reasoning", fontsize=12, fontweight="bold", pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(wino_methods, rotation=30)
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right", frameon=True)
        
        # Add labels helper
        def autolabel(rects):
            for rect in rects:
                h = rect.get_height()
                ax2.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + 1.0,
                    f"{h:.0f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight="bold"
                )
                
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
    else:
        ax2.text(0.5, 0.5, "Winoground Data Missing", ha='center', va='center', fontsize=12)
        
    # Global title
    fig.suptitle("Multimodal Compositionality & Visual Attribution Benchmark Suite", fontsize=15, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    output_image = os.path.join(results_dir, "multimodal_evaluation_results.png")
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to: {output_image}")

if __name__ == "__main__":
    main()
