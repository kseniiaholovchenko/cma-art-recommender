"""
Given an artwork index, return the top-k most similar artworks.
Input:  ../data/artworks_with_index.csv
        ../data/embeddings.npy
Output: ../data/recommender_artifact.pkl
        Prints top-10 recommendations to terminal.
"""

import os
import pickle
import numpy as np
import pandas as pd
from recommend import recommend

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


if __name__ == "__main__":
    index_path = os.path.join(DATA_DIR, "artworks_with_index.csv")
    embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")

    print(f"Loading {index_path}...")
    df = pd.read_csv(index_path)
    print(f"Loading {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"  {len(df)} artworks, embeddings shape: {embeddings.shape}")

    # Test on the first artwork; use embedding_index column, not bare positional 0
    query_artwork = df.iloc[0]
    query_idx = int(query_artwork["embedding_index"])
    print(f"\nQuery: '{query_artwork['title']}' by {query_artwork['artist']}")
    print("\nTop 10 recommendations:")

    results = recommend(query_idx, embeddings, df, k=10)
    for _, row in results.iterrows():
        print(f"  {row['similarity']:.3f}  {row['title']}  —  {row['artist']}")

    # Save artifact for notebooks; app.py loads embeddings.npy and
    # artworks_with_index.csv directly and does not use this file.
    artifact = {"embeddings": embeddings, "df": df}
    artifact_path = os.path.join(DATA_DIR, "recommender_artifact.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved artifact → {artifact_path}")
