"""
Build combined text per artwork, encode to vectors, save.
Input:  ../data/artworks_annotated.csv
Output: ../data/embeddings.npy
        ../data/artworks_with_index.csv
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def build_text_for_embedding(row):
    """
    Combine artwork fields into a single string for the embedding model.
    Field order: title, artist (skipped if 'Unknown'), medium, culture,
    type, description (scraped preferred if >50 chars), did_you_know,
    then mood/subject/style_period if annotated.
    Parts are joined with ". " so the model sees natural sentence boundaries.
    Parameters:
        row (pd.Series): One row from the artworks dataframe.
    Returns:
        str: Combined text ready for encoding.
    """
    # String columns are pre-filled with "" before this function is called
    # (see fillna in __main__). row.get() handles columns that are absent
    # entirely (returns None → falsy → ""). Do NOT rely on `x or ""` to
    # catch NaN — np.nan is truthy in Python, so `np.nan or ""` stays np.nan.
    parts = []

    title = str(row.get("title") or "").strip()
    if title:
        parts.append(title)

    # Skip artist if it contains "Unknown" — adds noise, not signal
    artist = str(row.get("artist") or "").strip()
    if artist and "Unknown" not in artist:
        parts.append(f"by {artist}")

    medium = str(row.get("medium") or "").strip()
    if medium:
        parts.append(medium)

    culture = str(row.get("culture") or "").strip()
    if culture:
        parts.append(culture)

    artwork_type = str(row.get("type") or "").strip()
    if artwork_type:
        parts.append(artwork_type)

    # Prefer scraped description (richer text) when long enough; fall back to
    # api_description, then use scraped even if short so no description is lost
    scraped = str(row.get("scraped_description") or "").strip()
    api_desc = str(row.get("api_description") or "").strip()
    if len(scraped) > 50:
        parts.append(scraped)
    elif api_desc:
        parts.append(api_desc)
    elif scraped:
        parts.append(scraped)

    did_you_know = str(row.get("did_you_know") or "").strip()
    if did_you_know:
        parts.append(did_you_know)

    # Add annotation labels if available (246 annotated paintings)
    mood = str(row.get("mood") or "").strip()
    subject = str(row.get("subject") or "").strip()
    style = str(row.get("style_period") or "").strip()
    if mood:
        parts.append(f"mood: {mood}")
    if subject:
        parts.append(f"subject: {subject}")
    if style:
        parts.append(f"style: {style}")


    return ". ".join(parts)


if __name__ == "__main__":
    clean_path = os.path.join(DATA_DIR, "artworks_annotated.csv")
    print(f"Loading {clean_path}...")
    df = pd.read_csv(clean_path)
    print(f"  Loaded {len(df)} artworks.")

    print("\nBuilding text_for_embedding column...")
    # np.nan is truthy in Python, so `np.nan or ""` stays np.nan and
    # str(np.nan) gives "nan". Fill string columns first to prevent
    # "mood: nan" / "subject: nan" leaking into embedding text for
    # unannotated rows.
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].fillna("")
    df["text_for_embedding"] = df.apply(build_text_for_embedding, axis=1)

    print("\nExample text for first artwork:")
    print(df["text_for_embedding"].iloc[0])

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    #model_name = "BAAI/bge-base-en-v1.5"
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = df["text_for_embedding"].tolist()
    print(f"Encoding {len(texts)} texts (batch_size=64)...")
    # normalize_embeddings=True: cosine similarity becomes a dot product (faster at query time)
    #tells the SentenceTransformer to divide each embedding vector by its L2 norm before returning it, so every vector stored in embeddings.npy already has length 1.
    embeddings = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )

    print(f"\nEmbedding shape: {embeddings.shape}")

    embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings → {embeddings_path}")

    # embedding_index lets app.py look up the correct row in the matrix
    df["embedding_index"] = range(len(df))
    index_path = os.path.join(DATA_DIR, "artworks_with_index.csv")
    df.to_csv(index_path, index=False)
    print(f"Saved artworks with index → {index_path}")
