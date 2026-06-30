"""
Collect raw artwork data from the Cleveland Museum of Art (CMA).
We use TWO data sources and combine them:
 1. CMA Open Access API  → structured metadata (title, artist, tags…)
 2. CMA website HTML     → richer description text not always in the API
OUTPUT  : data/artworks_raw.csv. One row per artwork, all fields raw (no cleaning done here).

!! FIRST RUN: keep LIMIT = 5 at the bottom to test selectors work.
       Check data/artworks_raw.csv to confirm scraped_description is not empty.
       Only then change LIMIT = 3000 for the full run.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from tqdm import tqdm  # shows a progress bar in the terminal

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# Base URL of the CMA Open Access API
BASE_API = "https://openaccess-api.clevelandart.org/api/artworks/"

# We identify ourselves in the User-Agent header.
# it tells the server this is academic research, not a malicious bot.
HEADERS = {
    "User-Agent": "Academic research scraper - BHT Berlin student project"
}

# Delay between HTML page requests (seconds).
# 0.3s = polite and unlikely to get blocked.
SCRAPE_DELAY = 0.3

# PART 1: FETCH FROM THE CMA API
def fetch_from_api(limit=1000):
    """
    Fetches paintings from the CMA API by looping through pages of 100.
    The API max is 100 per request, so we repeat until we have enough.

    Returns: list of raw artwork dicts from the API.
    """
    records = []
    params = {
        "cc0": 1,           # CC0 license only — free to use
        "has_image": 1,     # must have a photo
        "type": "Painting", # paintings only
        "limit": 100,       # max per request the API allows
        "skip": 0,          # increases by 100 each loop = next page
    }

    with tqdm(total=limit, desc="Fetching paintings from API") as pbar:
        while len(records) < limit:
            resp = requests.get(BASE_API, params=params, timeout=10)
            resp.raise_for_status()
            batch = resp.json().get("data", [])
            if not batch:
                break                        # no more results, stop
            before = len(records)
            records.extend(batch)
            params["skip"] += 100            # next page
            pbar.update(len(batch))
            time.sleep(0.15)

    return records[:limit] # ensure we return exactly the requested limit, in case we got a few extra on the last page


def parse_api_record(record):
    """
    Extracts the fields we care about from one raw API record.
    This is just field selection — no cleaning, no quality decisions.

    Key things to handle carefully:
    - creators: API returns a LIST of dicts, we take the first one
    - culture:  API returns a LIST e.g. ["France", "19th century"] — must join
    - images:   nested dict, we dig in to get the web-sized image URL
    - None values: many fields can be None, we default to "" to avoid NaN

    Parameters:
        record (dict): One raw JSON record from the API.

    Returns:
        dict: Flat dict with our selected fields, safe for pandas.
    """

    # --- Artist ---
    # "creators" is a list because some artworks have multiple creators.
    # We take the first one's "description" field.
    # If the list is empty, we use "Unknown".
    creators = record.get("creators") or []
    artist = creators[0].get("description", "Unknown") if creators else "Unknown"

    # --- Culture ---
    # IMPORTANT FIX: culture comes back as a LIST from the API,
    # e.g. ["France", "19th century"]
    # We join it into a single string: "France, 19th century"
    # Without this, pandas stores a Python list object in the cell,
    # which causes errors when the CSV is read back later.
    culture_raw = record.get("culture") or []
    if isinstance(culture_raw, list):
        culture = ", ".join(culture_raw)
    else:
        culture = str(culture_raw)

    # --- Image URL ---
    # The "images" field is a nested dict with different size options.
    # "web" is a good medium size for display in our Streamlit app.
    # We use chained .get() with defaults to safely handle missing keys.
    image_url = (record.get("images") or {}).get("web", {}).get("url", "")

    return {
        "id":               record.get("id"),
        "accession_number": record.get("accession_number", ""),
        "title":            record.get("title", ""),
        "artist":           artist,
        "date":             record.get("creation_date", ""),
        "medium":           record.get("technique", ""),
        "department":       record.get("department", ""),
        "culture":          culture,   # now always a string, never a list
        "type":             record.get("type", ""),
        # "api_description" is often short or missing.
        # We supplement it with scraped text from the website (see Part 2).
        "api_description":  record.get("description", "") or "",
        "did_you_know":     record.get("did_you_know", "") or "",
        "image_url":        image_url,
        "url":              record.get("url", ""),
    }


# PART 2: SCRAPE THE CMA WEBSITE FOR RICHER TEXT
def scrape_artwork_page(url):
    """
    Scrapes one artwork page and returns the curator description text.
    Uses <meta name="description"> tag — verified against live CMA site.
    """
    result = {"scraped_description": ""}
    if not url:
        return result
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return result
        soup = BeautifulSoup(response.text, "html.parser")
        meta_tag = soup.find("meta", {"name": "description"})
        if meta_tag and meta_tag.get("content"):
            result["scraped_description"] = meta_tag["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"  Warning: could not scrape {url}: {e}")
    return result


if __name__ == "__main__":
    LIMIT = 1200

    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: fetch from API
    print(f"Fetching {LIMIT} paintings from API...")
    records = fetch_from_api(limit=LIMIT)
    df = pd.DataFrame([parse_api_record(r) for r in records])
    print(f"Got {len(df)} records.\n")

    # Step 2: scrape each artwork page for richer description text
    print("Scraping artwork pages...")
    descriptions = []
    for url in tqdm(df["url"].tolist(), desc="Scraping"):
        descriptions.append(scrape_artwork_page(url)["scraped_description"])
        time.sleep(SCRAPE_DELAY)
    df["scraped_description"] = descriptions

    # Step 3: quick sanity check before saving
    filled = (df["scraped_description"] != "").sum()
    print(f"\nDescriptions scraped: {filled}/{len(df)}")
    if filled == 0:
        print("WARNING: all descriptions empty — check scrape_artwork_page()")
    else:
        print(f"Sample: {df['scraped_description'].iloc[0][:200]}")

    # Step 4: save raw — no cleaning here
    out_path = os.path.join(DATA_DIR, "artworks_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")