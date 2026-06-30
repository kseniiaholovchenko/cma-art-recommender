"""
Encode all artwork images with CLIP (openai/clip-vit-base-patch32).
Input:  ../data/artworks_with_index.csv  (has image_url column)
Output: ../data/image_embeddings_clip.npy  shape (n_artworks, 512)
        ../data/clip_index.csv             columns: id, embedding_index, image_url, clip_success

Artworks whose image download fails get a zero vector in the .npy file.
clip_index.csv records which rows succeeded so app.py can filter them.

Run on the cluster:
    kubectl apply -f src/job_clip_embed.yaml
"""

import io
import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
TIMEOUT = 10


def download_image(url):
    """Download image from URL, return PIL Image in RGB mode, or None on any error."""
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


if __name__ == "__main__":
    index_path = os.path.join(DATA_DIR, "artworks_with_index.csv")
    out_emb_path = os.path.join(DATA_DIR, "image_embeddings_clip.npy")
    out_idx_path = os.path.join(DATA_DIR, "clip_index.csv")

    print(f"Loading {index_path}...")
    df = pd.read_csv(index_path).reset_index(drop=True)
    n = len(df)
    print(f"  {n} artworks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: {MODEL_NAME}  (device: {device})")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    embeddings = np.zeros((n, 512), dtype=np.float32)
    clip_success = [False] * n
    urls = df["image_url"].fillna("").tolist()

    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="Encoding images"):
        batch_end = min(batch_start + BATCH_SIZE, n)

        images, valid_indices = [], []
        for i in range(batch_start, batch_end):
            if not urls[i]:
                continue
            img = download_image(urls[i])
            if img is not None:
                images.append(img)
                valid_indices.append(i)

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)
            features_np = features.cpu().numpy()

        for j, idx in enumerate(valid_indices):
            embeddings[idx] = features_np[j]
            clip_success[idx] = True

    np.save(out_emb_path, embeddings)
    print(f"Saved embeddings → {out_emb_path}  shape: {embeddings.shape}")

    id_col = df["id"] if "id" in df.columns else pd.RangeIndex(n)
    clip_df = pd.DataFrame({
        "id": id_col,
        "embedding_index": df.index,
        "image_url": urls,
        "clip_success": clip_success,
    })
    clip_df.to_csv(out_idx_path, index=False)
    print(f"Saved index    → {out_idx_path}")

    success_count = sum(clip_success)
    print(f"\nDone. {success_count}/{n} images encoded successfully "
          f"({n - success_count} failed/skipped).")
