import argparse
import sys
import os
import json
import torch
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Test Loading of MM-BRIGHT and ViDoRe datasets")
    parser.add_argument("--test-mm-bright", action="store_true", help="Load MM-BRIGHT examples")
    parser.add_argument("--test-vidore", action="store_true", help="Load a ViDoRe synthetic dataset")
    args = parser.parse_args()
    
    if args.test_mm_bright:
        print("Loading MM-BRIGHT ('examples' configuration, 'academia' split)...")
        try:
            ds = load_dataset("mm-bright/MM-BRIGHT", "examples", split="academia")
            print(f"Successfully loaded MM-BRIGHT! Number of samples: {len(ds)}")
            print("First example keys:", list(ds[0].keys()))
            print("Query text:", ds[0]["query"])
        except Exception as e:
            print(f"Failed to load MM-BRIGHT: {e}")
            
    if args.test_vidore:
        print("\nLoading ViDoRe Synthetic Energy dataset...")
        try:
            ds = load_dataset("vidore/syntheticDocQA_energy_test", split="test")
            print(f"Successfully loaded ViDoRe Synthetic! Number of samples: {len(ds)}")
            print("First example keys:", list(ds[0].keys()))
            print("Query text:", ds[0]["query"])
            print("Target image file:", ds[0]["image_filename"])
        except Exception as e:
            print(f"Failed to load ViDoRe Synthetic: {e}")

if __name__ == "__main__":
    main()
