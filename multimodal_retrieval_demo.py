import os
import math
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

def create_toy_image(shape, color, bg_color="white", size=(224, 224)):
    """Draw a geometric shape of a given color and save it as an image."""
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)
    w, h = size
    
    pad = 40
    if shape == "circle":
        draw.ellipse([pad, pad, w - pad, h - pad], fill=color, outline="black", width=2)
    elif shape == "square":
        draw.rectangle([pad, pad, w - pad, h - pad], fill=color, outline="black", width=2)
    elif shape == "triangle":
        points = [(w // 2, pad), (pad, h - pad), (w - pad, h - pad)]
        draw.polygon(points, fill=color, outline="black", width=2)
    elif shape == "star":
        cx, cy = w // 2, h // 2
        r_outer = 80
        r_inner = 35
        points = []
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = i * math.pi / 5 - math.pi / 2
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=color, outline="black", width=2)
    else:
        draw.rectangle([pad, pad + 20, w - pad, h - pad - 20], fill=color, outline="black", width=2)
        
    return img

def setup_dataset():
    """Setup a toy dataset of 10 different colored shapes."""
    output_dir = "data/multimodal_toy"
    os.makedirs(output_dir, exist_ok=True)
    
    toy_configs = [
        {"shape": "circle", "color": "red", "desc": "a red circle"},
        {"shape": "circle", "color": "blue", "desc": "a blue circle"},
        {"shape": "circle", "color": "green", "desc": "a green circle"},
        {"shape": "triangle", "color": "yellow", "desc": "a yellow triangle"},
        {"shape": "triangle", "color": "purple", "desc": "a purple triangle"},
        {"shape": "triangle", "color": "red", "desc": "a red triangle"},
        {"shape": "square", "color": "blue", "desc": "a blue square"},
        {"shape": "square", "color": "green", "desc": "a green square"},
        {"shape": "square", "color": "yellow", "desc": "a yellow square"},
        {"shape": "star", "color": "purple", "desc": "a purple star"}
    ]
    
    dataset = []
    print("\n--- Generating Toy Dataset Images ---")
    for idx, cfg in enumerate(toy_configs):
        img = create_toy_image(cfg["shape"], cfg["color"])
        img_path = os.path.join(output_dir, f"shape_{idx}.png")
        img.save(img_path)
        dataset.append({
            "id": idx,
            "path": img_path,
            "description": cfg["desc"],
            "shape": cfg["shape"],
            "color": cfg["color"]
        })
        print(f"  Generated shape_{idx}.png: {cfg['desc']}")
        
    return dataset

def extract_tensor(output):
    """Safely extract the raw tensor from CLIPModel outputs across different transformers versions."""
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (list, tuple)):
        return output[0]
    return output

def main():
    parser = argparse.ArgumentParser(description="Multimodal CLIP Retrieval Prototype")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model key")
    parser.add_argument("--query-text", type=str, default="a purple star", help="Text query for Text-to-Image retrieval")
    parser.add_argument("--query-shape", type=str, default="star", help="Shape of the query image for Image-to-Image retrieval")
    parser.add_argument("--query-color", type=str, default="blue", help="Color of the query image for Image-to-Image retrieval")
    args = parser.parse_args()
    
    # 1. Setup dataset
    dataset = setup_dataset()
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device.upper()}")
    
    # 2. Initialize CLIP
    from transformers import CLIPProcessor, CLIPModel
    print(f"Loading pretrained CLIP model: {args.model}...")
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    
    # 3. Compute database embeddings
    print("\n--- Indexing Database Images ---")
    image_embeddings = []
    for item in dataset:
        img = Image.open(item["path"])
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_embed = model.get_image_features(**inputs)
            img_embed = extract_tensor(img_embed)
            img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
            image_embeddings.append(img_embed.cpu().numpy()[0])
    print("Database image indexing completed.")
    
    # 4. Text-to-Image Retrieval Query
    print(f"\n==========================================")
    print(f"Executing Text-to-Image Query: '{args.query_text}'")
    print(f"==========================================")
    
    inputs_text = processor(text=[args.query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs_text)
        text_embed = extract_tensor(text_embed)
        text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
        text_embed = text_embed.cpu().numpy()[0]
        
    t2i_scores = []
    for idx, db_embed in enumerate(image_embeddings):
        sim = float(np.dot(text_embed, db_embed))
        t2i_scores.append((idx, sim))
    t2i_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop-3 retrieved results:")
    for rank, (idx, score) in enumerate(t2i_scores[:3]):
        db_item = dataset[idx]
        print(f"  Rank {rank+1}: {db_item['path']} (Description: {db_item['description']}) | Cosine Similarity = {score:.4f}")
        
    # 5. Image-to-Image Retrieval Query
    print(f"\n==========================================")
    print(f"Executing Image-to-Image Query: '{args.query_color} {args.query_shape}'")
    print(f"==========================================")
    
    # Generate a query image that might not even be in the database (e.g. blue star)
    query_dir = "data/multimodal_toy"
    query_img_path = os.path.join(query_dir, "query_image.png")
    query_img = create_toy_image(args.query_shape, args.query_color)
    query_img.save(query_img_path)
    print(f"Generated query image at: {query_img_path} ({args.query_color} {args.query_shape})")
    
    # Encode query image
    img = Image.open(query_img_path)
    inputs_img = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_query_embed = model.get_image_features(**inputs_img)
        img_query_embed = extract_tensor(img_query_embed)
        img_query_embed = img_query_embed / img_query_embed.norm(p=2, dim=-1, keepdim=True)
        img_query_embed = img_query_embed.cpu().numpy()[0]
        
    i2i_scores = []
    for idx, db_embed in enumerate(image_embeddings):
        sim = float(np.dot(img_query_embed, db_embed))
        i2i_scores.append((idx, sim))
    i2i_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop-3 retrieved results:")
    for rank, (idx, score) in enumerate(i2i_scores[:3]):
        db_item = dataset[idx]
        print(f"  Rank {rank+1}: {db_item['path']} (Description: {db_item['description']}) | Cosine Similarity = {score:.4f}")
        
    # Concept Expansion Simulation (VLM-style)
    print(f"\n==========================================")
    print(f"Simulating VLM-Augmented Query Expansion")
    print(f"==========================================")
    vlm_caption = f"A detailed 2D outline of a sharp {args.query_color} {args.query_shape} centered on a white canvas."
    print(f"VLM generated caption for query image: '{vlm_caption}'")
    
    inputs_vlm = processor(text=[vlm_caption], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vlm_embed = model.get_text_features(**inputs_vlm)
        vlm_embed = extract_tensor(vlm_embed)
        vlm_embed = vlm_embed / vlm_embed.norm(p=2, dim=-1, keepdim=True)
        vlm_embed = vlm_embed.cpu().numpy()[0]
        
    vlm_scores = []
    for idx, db_embed in enumerate(image_embeddings):
        sim = float(np.dot(vlm_embed, db_embed))
        vlm_scores.append((idx, sim))
    vlm_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop-3 retrieved results using VLM expanded caption:")
    for rank, (idx, score) in enumerate(vlm_scores[:3]):
        db_item = dataset[idx]
        print(f"  Rank {rank+1}: {db_item['path']} (Description: {db_item['description']}) | Cosine Similarity = {score:.4f}")

if __name__ == "__main__":
    main()
