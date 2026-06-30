"""
Shared recommendation logic used by 5_recommender.py and app.py.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

#artwork-to-artwork: user picks a painting, finds similar ones. Excludes the query itself from results.
def recommend(query_idx, embeddings, df, k=10):
    """
    Return the top-k artworks most similar to the query artwork.
    Embeddings are already L2-normalized (from 3_embed.py), so cosine
    similarity equals the dot product — no extra normalization needed here.

    Parameters:
        query_idx (int): The embedding_index value for the query artwork
                         (equals the positional row in both df and embeddings).
        embeddings (np.ndarray): Normalized embedding matrix (n_artworks, dim),
                                 aligned row-for-row with df.
        df (pd.DataFrame): Full, unfiltered artworks dataframe so that
                           positional index == embedding_index.
        k (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: Top-k similar rows from df with a "similarity" column added.
    """
    assert len(df) == len(embeddings), (
        f"df ({len(df)} rows) and embeddings ({len(embeddings)} rows) are misaligned — "
        "pass the full, unfiltered dataframe."
    )
    # sklearn.metrics.pairwise.cosine_similarity. Internally, sklearn computes cosine similarity as:
    #cosine(A, B) = (A · B^T) / (||A|| * ||B||)
    #The A · B^T part is a matrix multiplication. 
    #In my case, since the embeddings are already L2-normalized, the division becomes trivial (/ 1.0), so it reduces to a pure dot product — which is still a matrix multiply under the hood.
    query_vec = embeddings[query_idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]
    sims[query_idx] = -1  # exclude the query artwork itself
    top_indices = np.argsort(sims)[::-1][:k]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices]
    return results


#ree text query: user types "calm Japanese landscape", gets encoded to a vector, finds similar paintings. 
#No self-exclusion needed.
def recommend_by_vector(query_vec, embeddings, df, k=10):
    """
    Return the top-k artworks most similar to an arbitrary query vector.
    Used for free-text queries: the caller encodes the text with the same
    SentenceTransformer model used in 3_embed.py, then passes the vector here.
    No self-exclusion is applied because the query is not in the dataset.

    Parameters:
        query_vec (np.ndarray): 1-D encoded query vector (will be L2-normalized).
        embeddings (np.ndarray): Normalized embedding matrix (n_artworks, dim),
                                 aligned row-for-row with df.
        df (pd.DataFrame): Full, unfiltered artworks dataframe.
        k (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: Top-k similar rows from df with a "similarity" column added.
    """
    assert len(df) == len(embeddings), (
        f"df ({len(df)} rows) and embeddings ({len(embeddings)} rows) are misaligned — "
        "pass the full, unfiltered dataframe."
    )
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    # Normalize so cosine similarity is comparable to the stored L2-normalized vectors
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    sims = cosine_similarity(q, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices]
    return results


def recommend_by_clip_text(text_query, clip_embeddings, df, clip_model, clip_processor, k=10):
    """
    Encode a text query with CLIP's text encoder and find visually similar paintings.
    Uses CLIP's joint text-image space — the text vector is compared directly against
    image vectors, so results reflect visual similarity to the description.

    Parameters:
        text_query (str): Free-text visual description, e.g. "red dramatic painting with horses".
        clip_embeddings (np.ndarray): L2-normalized CLIP image embeddings (n_artworks, 512).
        df (pd.DataFrame): Full artworks dataframe aligned row-for-row with clip_embeddings.
                           If a "clip_success" column is present, failed rows are excluded.
        clip_model: Loaded CLIPModel (caller caches with @st.cache_resource).
        clip_processor: Loaded CLIPProcessor (caller caches with @st.cache_resource).
        k (int): Number of results to return.

    Returns:
        pd.DataFrame: Top-k rows from df with a "similarity" column added.
    """
    assert len(df) == len(clip_embeddings), (
        f"df ({len(df)} rows) and clip_embeddings ({len(clip_embeddings)} rows) are misaligned."
    )
    clip_model.eval()
    with torch.no_grad():
        inputs = clip_processor(
            text=[text_query], return_tensors="pt", padding=True, truncation=True
        )
        features = clip_model.get_text_features(**inputs)
        features = F.normalize(features, dim=-1)
        query_vec = features.cpu().numpy()[0]

    q = query_vec.reshape(1, -1).astype(np.float32)
    sims = cosine_similarity(q, clip_embeddings)[0]

    if "clip_success" in df.columns:
        sims[~df["clip_success"].values.astype(bool)] = -1.0

    top_indices = np.argsort(sims)[::-1][:k]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices]
    return results


def recommend_by_clip_image(pil_image, clip_embeddings, df, clip_model, clip_processor, k=10):
    """
    Encode an uploaded PIL image with CLIP's image encoder and find visually similar paintings.

    Parameters:
        pil_image (PIL.Image): The uploaded image (any mode — processor handles conversion).
        clip_embeddings (np.ndarray): L2-normalized CLIP image embeddings (n_artworks, 512).
        df (pd.DataFrame): Full artworks dataframe aligned row-for-row with clip_embeddings.
                           If a "clip_success" column is present, failed rows are excluded.
        clip_model: Loaded CLIPModel (caller caches with @st.cache_resource).
        clip_processor: Loaded CLIPProcessor (caller caches with @st.cache_resource).
        k (int): Number of results to return.

    Returns:
        pd.DataFrame: Top-k rows from df with a "similarity" column added.
    """
    assert len(df) == len(clip_embeddings), (
        f"df ({len(df)} rows) and clip_embeddings ({len(clip_embeddings)} rows) are misaligned."
    )
    clip_model.eval()
    with torch.no_grad():
        inputs = clip_processor(images=pil_image, return_tensors="pt")
        features = clip_model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1)
        query_vec = features.cpu().numpy()[0]

    q = query_vec.reshape(1, -1).astype(np.float32)
    sims = cosine_similarity(q, clip_embeddings)[0]

    if "clip_success" in df.columns:
        sims[~df["clip_success"].values.astype(bool)] = -1.0

    top_indices = np.argsort(sims)[::-1][:k]
    results = df.iloc[top_indices].copy()
    results["similarity"] = sims[top_indices]
    return results
