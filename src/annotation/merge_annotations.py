"""
Merge artworks with Label Studio annotations.
Input:  <project_root>/data/artworks_clean.csv
        <project_root>/src/annotation/annotations_clean.csv
Output: <project_root>/data/artworks_annotated.csv

Merge is a LEFT JOIN on "id" — all artworks are kept, only annotated rows
get mood/subject/style_period values. Unannotated rows get NaN in those columns.
Run 3_embed.py next to build text_for_embedding from the merged data.
"""

import os
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
ANNOTATION_DIR = os.path.dirname(__file__)


if __name__ == "__main__":
    clean_path = os.path.join(DATA_DIR, "artworks_clean.csv")
    annotations_path = os.path.join(ANNOTATION_DIR, "annotations_clean.csv")

    print(f"Loading {clean_path}...")
    df_artworks = pd.read_csv(clean_path)
    print(f"  Loaded {len(df_artworks)} artworks.")

    print(f"Loading {annotations_path}...")
    df_annotations = pd.read_csv(annotations_path)
    df_annotations = df_annotations.drop_duplicates(subset=["id"], keep="last")
    print(f"  Loaded {len(df_annotations)} annotations.")

    df_merged = df_artworks.merge(df_annotations, on="id", how="left")
    if len(df_merged) != len(df_artworks):
        raise ValueError(
            f"Merge produced {len(df_merged)} rows but input had {len(df_artworks)}. "
            "Duplicate artwork IDs in annotations_clean.csv."
        )
    print(f"  Merged: {len(df_merged)} rows.")

    annotated_count = int(df_merged["mood"].notna().sum())
    print(f"  Annotated: {annotated_count} / {len(df_merged)} artworks")

    output_path = os.path.join(DATA_DIR, "artworks_annotated.csv")
    df_merged.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
