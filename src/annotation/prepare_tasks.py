"""
Sample artworks from the clean dataset and format them for Label Studio.
Input:  <project_root>/data/artworks_clean.csv
Output: <project_root>/src/annotation/tasks.json
"""

import os
import json
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
ANNOTATION_DIR = os.path.dirname(__file__)


def build_task(row):
    """
    Convert one artwork row into a Label Studio task dict.
    Parameters:
        row (pd.Series): One row from the artworks dataframe.

    Returns:
        dict: {"id": int, "data": {...}} in Label Studio import format.
    """
    scraped = "" if pd.isna(row.get("scraped_description")) else str(row.get("scraped_description"))
    api_desc = "" if pd.isna(row.get("api_description")) else str(row.get("api_description"))
    desc = scraped.strip() if scraped.strip() else api_desc

    def _str(val):
        return "" if pd.isna(val) else str(val)

    return {
        "data": {
            "id":          int(row["id"]) if pd.notna(row.get("id")) else None,
            "image":       _str(row.get("image_url")),
            "title":       _str(row.get("title")),
            "artist":      _str(row.get("artist")),
            "medium":      _str(row.get("medium")),
            "date":        _str(row.get("date")),
            "description": desc[:300],
        },
    }


if __name__ == "__main__":
    clean_path = os.path.join(DATA_DIR, "artworks_clean.csv")
    print(f"Loading {clean_path}...")
    df = pd.read_csv(clean_path)
    print(f"  Loaded {len(df)} artworks.")

    sample_size = min(250, len(df))
    sample = df.sample(n=sample_size, random_state=42)
    print(f"  Sampled {sample_size} artworks.")

    tasks = [build_task(row) for _, row in sample.iterrows()]

    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    tasks_path = os.path.join(ANNOTATION_DIR, "tasks.json")
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(tasks)} tasks → {tasks_path}")
    print("Import tasks.json into Label Studio to start annotating.")
