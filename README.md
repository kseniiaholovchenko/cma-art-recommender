# Art recommender - Cleveland Museum of Art

A semantic and visual art recommender for the Cleveland Museum of Art (CMA) collection. 

This Streamlit application allows users to explore the museum's artworks using natural language descriptions and visual similarities.

## Features
* **Artwork to Artwork:** Discover visually and conceptually related paintings starting from a piece you already love.
* **Text to Artwork:** Use natural language (e.g., "stormy sea," "19th century portrait") to semantically search the collection using a MiniLM text transformer.
* **Image to Artwork (Visual Search):** Upload your own image or describe a visual scene to find pixel-level similarities using OpenAI's CLIP vision-language model.

## Tech Stack
* **Frontend:** Streamlit
* **Models:** `all-MiniLM-L6-v2` (SentenceTransformers), `CLIP ViT-B/32`

## Dataset & Preprocessing

The recommendation engine is powered by a custom dataset built by combining two sources from the Cleveland Museum of Art to ensure maximum data richness:

1. **CMA Open Access API:** Provides the foundational, structured metadata (Title, Artist, Medium, Tags, etc.).
2. **Web Scraping (CMA Website HTML):** Extracts the richer, curatorial description text directly from the museum's website, as description text is not always in the API.

To ensure high-quality embeddings, the raw data undergoes a preprocessing pipeline:

* **Initial Raw Records:** 1,200 artworks
* **Final Cleaned Dataset:** 1,154 artworks
* **Data Quality Actions:** Removed 39 duplicates and 7 records missing descriptions.
* **Completeness:** Core metadata (title, artist, medium, and image_url) sits at 100% completeness post-cleaning. 

*Note: The dataset contains 377 pieces attributed to "Unknown artist." The semantic text and CLIP visual models handle these gracefully by relying on stylistic and descriptive vectors rather than explicit artist metadata.*
