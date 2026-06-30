"""
Measure data quality issues, fix them, save clean data.
Input:  ../data/artworks_raw.csv
Output: ../data/artworks_clean.csv
        ../data/quality_report.json
"""

import os
import json
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def measure_completeness(df, fields):
    """
    Compute completeness (% of non-null, non-empty rows) for each field.
    Parameters:
        df (pd.DataFrame): Dataframe to measure.
        fields (list): Column names to check.
    Returns:
        dict: {field_name: completeness_pct} rounded to 2 decimal places.
    """
    result = {}
    for field in fields:
        if field not in df.columns:
            result[field] = 0.0
            continue
        # Complete = not null AND not empty/whitespace string
        non_empty = df[field].apply(
            lambda x: str(x).strip() not in ("", "nan")
        )
        result[field] = round(non_empty.sum() / len(df) * 100, 2) if len(df) > 0 else 0.0
    return result


def count_unknown_artists(df):
    #Return number of rows where artist field contains the word 'Unknown'.
    return int(df["artist"].str.contains("Unknown", na=False).sum())


def count_duplicates(df):
    #Return number of duplicate rows based on (title, artist) pair.
    return int(df.duplicated(subset=["title", "artist"]).sum())


def count_missing_image(df):
    #Return number of rows where image_url is null or empty string.
    return int((df["image_url"].isnull() | (df["image_url"].fillna("").str.strip() == "")).sum())


def count_missing_description(df):
    #Return number of rows where BOTH api_description and scraped_description are empty.
    api_col = "api_description"
    scraped_col = "scraped_description"

    if api_col not in df.columns or scraped_col not in df.columns:
        df = df.copy()
        if api_col not in df.columns:
            df[api_col] = ""
        if scraped_col not in df.columns:
            df[scraped_col] = ""

    api_empty = df[api_col].isnull() | (df[api_col].fillna("").str.strip() == "")
    scraped_empty = df[scraped_col].isnull() | (df[scraped_col].fillna("").str.strip() == "")
    return int((api_empty & scraped_empty).sum())


if __name__ == "__main__":
    raw_path = os.path.join(DATA_DIR, "artworks_raw.csv")
    print(f"Loading {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"  Loaded {len(df)} rows.")

    total_raw = len(df)

    # Fields to measure completeness on (BEFORE any cleaning)
    completeness_fields = ["title", "artist", "medium", "api_description", "scraped_description", "image_url"]

    print("\nMeasuring completeness before cleaning...")
    completeness_before = measure_completeness(df, completeness_fields)
    for field, pct in completeness_before.items():
        print(f"  {field}: {pct}%")

    # Count known issues
    unknown_artist_count = count_unknown_artists(df)
    dup_count = count_duplicates(df)
    missing_image_count = count_missing_image(df)
    missing_desc_count = count_missing_description(df)

    print(f"\nKnown issues:")
    print(f"  Rows with 'Unknown' artist:         {unknown_artist_count}")
    print(f"  Duplicate rows (title+artist):       {dup_count}")
    print(f"  Rows with missing image_url:         {missing_image_count}")
    print(f"  Rows missing both descriptions:      {missing_desc_count}")

    # Fix 1: drop duplicates — keep first occurrence
    df = df.drop_duplicates(subset=["title", "artist"], keep="first")
    removed_duplicates = total_raw - len(df)
    print(f"\nAfter removing duplicates: {len(df)} rows (removed {removed_duplicates})")

    # Fix 2: drop rows with null or empty image_url
    before_image_drop = len(df)
    df = df[df["image_url"].notna() & (df["image_url"].fillna("").str.strip() != "")]
    removed_missing_image = before_image_drop - len(df)
    print(f"After removing missing images: {len(df)} rows (removed {removed_missing_image})")

    # Fix 3: drop rows with no usable description text (both sources empty)
    before_desc_drop = len(df)
    api_empty = df["api_description"].fillna("").str.strip() == ""
    scraped_empty = df["scraped_description"].fillna("").str.strip() == ""
    df = df[~(api_empty & scraped_empty)]
    removed_missing_desc = before_desc_drop - len(df)
    print(f"After removing missing descriptions: {len(df)} rows (removed {removed_missing_desc})")

    total_after = len(df)

    # Save quality report
    report = {
        "total_raw": total_raw,
        "total_after_cleaning": total_after,
        "removed_duplicates": removed_duplicates,
        "removed_missing_image": removed_missing_image,
        "removed_missing_desc": removed_missing_desc,
        "completeness_before_cleaning": completeness_before,
        "unknown_artist_count": unknown_artist_count,
        "missing_description_count": missing_desc_count,
    }

    report_path = os.path.join(DATA_DIR, "quality_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved quality report → {report_path}")

    # Save clean data — no new columns, no embedding text here
    clean_path = os.path.join(DATA_DIR, "artworks_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"Saved {total_after} artworks → {clean_path}")
