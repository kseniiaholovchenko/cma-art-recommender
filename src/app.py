"""
Cleveland Museum of Art — Semantic Art Recommender

Run from the src/ folder:
    streamlit run app_redesign_v2.py

Recommendation logic is unchanged:
- Similar-artwork mode calls recommend(...)
- Free-text mode encodes the query and calls recommend_by_vector(...)
"""

import os
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from recommend import (
    recommend,
    recommend_by_vector,
    recommend_by_clip_text,
    recommend_by_clip_image,
)


DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
)


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "artworks_with_index.csv"))
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    return df, embeddings


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data
def load_clip_embeddings():
    """Return CLIP image embeddings array, or None if not yet generated."""
    path = os.path.join(DATA_DIR, "image_embeddings_clip.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


@st.cache_data
def load_clip_index():
    """Return clip_index DataFrame (has clip_success column), or None."""
    path = os.path.join(DATA_DIR, "clip_index.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_clip_model():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Libre+Caslon+Display&display=swap');

:root {
    --bg: #F4EFE8;
    --paper: #FCFAF7;
    --paper-2: #F8F3ED;
    --ink: #181513;
    --muted: #6D625A;
    --line: #E4D8CC;
    --line-2: #D7C7B7;
    --accent: #7A3148;
    --accent-2: #AD8B5D;
    --accent-soft: #F5EAEE;
    --shadow: 0 18px 50px rgba(35, 22, 13, 0.07);
}

#MainMenu, header, footer { visibility: hidden; }
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(173,139,93,0.12), transparent 26%),
        linear-gradient(180deg, #FAF6F1 0%, var(--bg) 100%);
}
[data-testid="stMainBlockContainer"] {
    max-width: 1580px;
    padding: 24px 34px 54px;
}
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding: 14px 12px 30px; }
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
    color: var(--ink);
}

h1, h2, h3, h4, 
.page-title, 
.section-title, 
.artwork-title, 
.card-title {
    font-family: "Libre Caslon Display", serif !important;
    font-weight: normal !important;
}

[data-testid="stTextInput"] input,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 52px;
    height: 100%;
    box-sizing: border-box;
    background: rgba(255,255,255,0.9);
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: none;
    color: var(--ink);
}
/* BaseWeb wraps the <input> in its own div with a fixed height + overflow:hidden,
   which clips a 52px-tall input down to its default ~38px — let it grow instead. */
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] [data-baseweb="base-input"] {
    height: 100% !important;
    min-height: 52px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(122,49,72,0.10);
}
[data-testid="stTextInput"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    color: var(--muted);
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    overflow: visible !important;
    white-space: nowrap !important;
    max-width: none !important;
    flex-shrink: 0 !important;
    min-width: fit-content !important;
}

.stButton > button {
    min-height: 52px;
    box-sizing: border-box;
    border-radius: 14px;
    padding: 0.62rem 1rem;
    font-size: 0.88rem;
    font-weight: 700;
    transition: all 0.18s ease;
}
.stButton > button[kind="primary"] {
    background: var(--accent);
    border: 1px solid var(--accent);
    color: white;
}
.stButton > button[kind="primary"]:hover {
    background: #63273A;
    border-color: #63273A;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.84);
    border: 1px solid var(--line-2);
    color: var(--ink);
}
.stButton > button[kind="secondary"]:hover {
    background: var(--accent-soft);
    color: var(--accent);
    border-color: rgba(122,49,72,0.24);
}

[data-testid="stRadio"] [role="radiogroup"] {
    gap: 10px;
}
[data-testid="stRadio"] label[data-baseweb="radio"] {
    width: auto;
    margin: 0;
    padding: 0.62rem 0.95rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(255,255,255,0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: var(--ink);
    border-color: var(--ink);
    color: white;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
    color: white !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
[data-testid="stRadio"] label[data-baseweb="radio"] p {
    font-size: 0.86rem;
    font-weight: 700;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--line);
    border-radius: 26px;
    background: rgba(252,250,247,0.92);
    box-shadow: var(--shadow);
}

/* Custom layout blocks */
.top-shell {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.9fr);
    gap: 18px;
    margin-bottom: 24px;
}
.title-panel {
    padding: 26px 28px;
    border: 1px solid var(--line);
    border-radius: 28px;
    background: rgba(252,250,247,0.9);
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.info-panel {
    padding: 26px 28px;
    border: 1px solid var(--accent);
    border-radius: 28px;
    background: var(--accent); /* This is your button color */
    box-shadow: var(--shadow);
}
.kicker {
    color: var(--accent);
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.page-title {
    margin: 10px 0 12px;
    font-size: clamp(2.7rem, 4.2vw, 5rem);
    line-height: 0.94;
    color: var(--ink);
}
.page-copy {
    max-width: 760px;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
}
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
}
.meta-pill {
    padding: 0.58rem 0.8rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(255,255,255,0.78);
    color: var(--ink);
    font-size: 0.76rem;
    font-weight: 700;
}
.info-panel {
    display: grid;
    align-content: space-between;
    gap: 16px;
}
.info-card {
    padding: 16px 18px;
    border: 1px solid var(--line);
    border-radius: 20px;
    background: var(--paper);
}
.info-label {
    color: var(--muted);
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}
.info-value {
    margin-top: 8px;
    color: var(--ink);
    font-family: "Libre Caslon Display", serif;
    font-size: 2rem;
    line-height: 1;
}
.info-copy {
    margin-top: 6px;
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.6;
}

.section-label {
    margin: 2px 0 4px;
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.section-title {
    margin: 0 0 6px;
    color: var(--ink);
    font-size: 2.2rem;
    line-height: 1.02;
}
.section-copy {
    margin: 0;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.7;
}

.artwork-image,
.recommendation-image {
    width: 100%;
    display: block;
    object-fit: contain;
    background: linear-gradient(180deg, #F4ECE4 0%, #EFE4DA 100%);
}
.artwork-image {
    max-height: 880px;
    border-radius: 20px 20px 0 0;
}
.recommendation-image {
    height: 380px;
    border-radius: 18px;
}
.image-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    background: linear-gradient(180deg, #F4ECE4 0%, #EFE4DA 100%);
    color: var(--muted);
    border-radius: 18px;
}

.detail-shell {
    padding: 0 4px 6px;
}
.artwork-title {
    margin: 18px 0 8px;
    color: var(--ink);
    font-size: 2.5rem;
    line-height: 0.98;
}
.artwork-artist {
    margin: 0;
    color: var(--accent);
    font-size: 0.98rem;
    font-weight: 800;
}
.artwork-meta {
    margin-top: 10px;
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.6;
}
.artwork-description {
    margin-top: 18px;
    color: #4E4540;
    font-size: 0.94rem;
    line-height: 1.85;
}
.cma-link {
    display: inline-block;
    margin-top: 20px;
    color: var(--ink) !important;
    text-decoration: none !important;
    border-bottom: 1px solid var(--accent-2);
    font-size: 0.77rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.recs-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.recs-chip {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 0 0.75rem;
    border-radius: 999px;
    background: rgba(245,234,238,0.9);
    color: var(--accent);
    font-size: 0.74rem;
    font-weight: 800;
}
.card-pad { padding: 6px 4px 2px; }
.card-title {
    margin: 14px 0 6px;
    min-height: 2.3em;
    color: var(--ink);
    font-family: "Libre Caslon Display", serif;
    font-size: 1.6rem;
    line-height: 1.08;
}
.card-artist {
    overflow: hidden;
    margin-bottom: 14px;
    color: var(--muted);
    font-size: 1rem;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.score-badge {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 0.72rem;
    border-radius: 999px;
    background: rgba(245,234,238,0.9);
    color: var(--accent);
    font-size: 0.73rem;
    font-weight: 800;
}
.card-link {
    color: var(--muted) !important;
    font-size: 0.73rem;
    font-weight: 800;
    text-decoration: none !important;
}

.query-callout {
    margin-bottom: 18px;
    padding: 16px 18px;
    border: 1px solid rgba(122,49,72,0.14);
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(245,234,238,0.88) 0%, rgba(255,255,255,0.74) 100%);
    color: #503745;
    font-size: 0.9rem;
    line-height: 1.55;
}
.empty-state {
    padding: 74px 24px;
    border: 1px dashed rgba(173,139,93,0.48);
    border-radius: 26px;
    background: rgba(255,255,255,0.56);
    color: var(--muted);
    text-align: center;
}
.empty-state strong {
    display: block;
    margin-bottom: 10px;
    color: var(--ink);
    font-family: "Libre Caslon Display", serif;
    font-size: 2rem;
}

/* Number input: stNumberInputContainer is the real wrapper around BOTH the
   input and the step buttons (confirmed from Streamlit source) — round and
   border THIS as one pill, with overflow:hidden clipping the step buttons
   to follow its shape, instead of rounding each piece separately. */
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"],
[data-testid="stNumberInput"] [data-testid="stNumberInputContainer"].focused {
    height: 100% !important;
    min-height: 52px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: stretch !important;
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    outline: none !important;
    overflow: hidden !important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputField"],
[data-testid="stNumberInput"] input {
    height: 100%;
    min-height: unset !important;
    box-sizing: border-box;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--ink);
    text-align: center;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] {
    height: 100% !important;
    border: none !important;
    border-left: 1px solid var(--line) !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    background: rgba(255,255,255,0.5) !important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"]:hover,
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"]:hover {
    color: var(--ink) !important;
    background: var(--paper-2) !important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputIcon"] {
    color: var(--muted) !important;
    fill: var(--muted) !important;
}
[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"]:hover [data-testid="stNumberInputIcon"],
[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"]:hover [data-testid="stNumberInputIcon"] {
    color: var(--ink) !important;
    fill: var(--ink) !important;
}

/* Selectbox: left-aligned text, vertically centred */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    align-items: center;
}

.card-img-wrapper {
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 24px !important;
}
.card-img-wrapper img {
    display: block;
    width: 100%;
    border-radius: 18px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.card-img-wrapper) [data-testid="stButton"] {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.card-img-wrapper) [data-testid="stButton"] > button {
    width: 100% !important;
    background: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 999px !important;
    min-height: unset !important;
    padding: 0.62rem 1rem !important;
    cursor: pointer !important;
    transition: background 0.18s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.card-img-wrapper) [data-testid="stButton"] > button:hover {
    background: #383230 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.card-img-wrapper) [data-testid="stButton"] > button p {
    margin: 0;
    color: white;
    text-align: center;
    font-family: "Inter", sans-serif;
    font-size: 0.88rem !important;
    font-weight: 700;
    letter-spacing: 0.07em;
    white-space: nowrap;
}

@media (max-width: 1040px) {
    .top-shell { grid-template-columns: 1fr; }
    .artwork-image { max-height: 700px; }
}
@media (max-width: 760px) {
    .title-panel, .info-panel { padding: 20px 18px; border-radius: 20px; }
    .page-title { font-size: 2.45rem; }
    .section-title { font-size: 1.85rem; }
    .artwork-title { font-size: 2rem; }
    .recommendation-image { height: 280px; }
}

/* 1. Strip the default bluish background from inner input wrappers */
[data-testid="stNumberInput"] [data-baseweb="base-input"],
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="input"] > div {
    background-color: transparent !important;
}

/* 2. Force text inside the radio pills to perfectly center */
[data-testid="stRadio"] label[data-baseweb="radio"] p {
    width: 100% !important;
    text-align: center !important;
    margin: 0 !important;
}
</style>
"""


def value_or(value, fallback=""):
    if value is None or pd.isna(value):
        return fallback
    value = str(value).strip()
    return value if value else fallback


def image_markup(image_url, css_class, alt_text, fallback_height=None):
    image_url = value_or(image_url)
    alt_text = escape(alt_text)
    if image_url:
        return (
            f'<img class="{css_class}" src="{escape(image_url, quote=True)}" '
            f'alt="{alt_text}" loading="lazy">'
        )

    min_height = f" style='min-height:{fallback_height}px;'" if fallback_height else ""
    return f'<div class="image-fallback"{min_height}>Image unavailable</div>'


def open_artwork_as_query(embedding_index):
    st.session_state["pinned_idx"] = int(embedding_index)
    st.session_state["app_mode"] = "Browse by artwork"


def set_example_query(query):
    st.session_state["text_query_input"] = query
    st.session_state["text_query"] = query


def render_top_shell(artwork_count):
    st.markdown(
        f"""
        <section class="top-shell">
            <div class="title-panel">
                <div class="kicker">Cleveland Museum of Art · Semantic Explorer</div>
                <h1 class="page-title">A richer way to browse paintings.</h1>
                <p class="page-copy">
                    Explore the Cleveland Museum of Art's collection through the lens of artificial intelligence. Uncover hidden connections between artworks using natural language descriptions, visual similarities, or by starting with a painting you already love.
                </p>
            </div>
            <div class="info-panel">
                <div class="info-card">
                    <div class="info-label">Collection</div>
                    <div class="info-value">{artwork_count:,}</div>
                    <div class="info-copy">Paintings available for recommendation and semantic search.</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Models</div>
                    <div class="info-value">Two main models</div>
                    <div class="info-copy">The recommender uses two models: MiniLM (text transformer - 384-dimensional text vectors) and CLIP (vision-language model - 512-dimensional image vectors).</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(label, title, copy):
    st.markdown(
        f"""
        <div class="section-label">{escape(label)}</div>
        <h2 class="section-title">{escape(title)}</h2>
        <p class="section-copy">{escape(copy)}</p>
        """,
        unsafe_allow_html=True,
    )


def render_artwork_details(row, label=None):
    if label:
        st.markdown(f'<div class="section-label">{escape(label)}</div>', unsafe_allow_html=True)

    title = value_or(row.get("title"), "Untitled")
    artist = value_or(row.get("artist"), "Unknown artist")
    date = value_or(row.get("date"))
    medium = value_or(row.get("medium"))
    department = value_or(row.get("department"))
    metadata = " · ".join(item for item in (date, medium, department) if item)

    st.markdown('<div class="detail-shell">', unsafe_allow_html=True)
    st.markdown(
        image_markup(row.get("image_url"), "artwork-image", title, fallback_height=520),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <h2 class="artwork-title">{escape(title)}</h2>
        <p class="artwork-artist">{escape(artist)}</p>
        {f'<div class="artwork-meta">{escape(metadata)}</div>' if metadata else ''}
        """,
        unsafe_allow_html=True,
    )

    description = value_or(row.get("scraped_description")) or value_or(row.get("api_description"))
    if description:
        shortened = description[:700] + ("…" if len(description) > 700 else "")
        st.markdown(
            f'<div class="artwork-description">{escape(shortened)}</div>',
            unsafe_allow_html=True,
        )

    url = value_or(row.get("url"))
    if url:
        st.markdown(
            f'<a class="cma-link" href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
            'View at Cleveland Museum of Art ↗</a>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_recommendation_card(rec, card_key):
    title = value_or(rec.get("title"), "Untitled")
    artist = value_or(rec.get("artist"), "Unknown artist")
    url = value_or(rec.get("url"))
    similarity = float(rec.get("similarity", 0.0))
    embedding_index = rec.get("embedding_index", -1)

    with st.container(border=True):
        # 1. Image block
        st.markdown(
            f'<div class="card-pad" style="padding-bottom: 0;">'
            f'<div class="card-img-wrapper">'
            + image_markup(rec.get("image_url"), "recommendation-image", title, fallback_height=380)
            + '</div></div>',
            unsafe_allow_html=True,
        )
        
        # 2. Button
        if pd.notna(embedding_index) and int(embedding_index) >= 0:
            st.button(
                "Find similar",
                key=f"explore_{card_key}",
                on_click=open_artwork_as_query,
                args=(int(embedding_index),),
                use_container_width=True,
            )
            
        # 3. Text block (Flexbox to push the footer down, maintaining card height)
        st.markdown(
            f"""
            <div class="card-pad" style="padding-top: 0; padding-bottom: 20px; margin-top: -4px; display: flex; flex-direction: column; min-height: 140px;">
                <div class="card-title" style="margin-top: 0; min-height: 0;" title="{escape(title, quote=True)}">{escape(title)}</div>
                <div class="card-artist" title="{escape(artist, quote=True)}">{escape(artist)}</div>
                <div class="card-footer" style="margin-top: auto;">
                    <span class="score-badge">Similarity {similarity:.3f}</span>
                    {f'<a class="card-link" href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">CMA ↗</a>' if url else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_recommendations_grid(recs, key_prefix="rec"):
    if recs.empty:
        return
    for start in range(0, len(recs), 2):
        cols = st.columns(2, gap="large")
        row_slice = recs.iloc[start:start + 2]
        for col, (idx, rec) in zip(cols, row_slice.iterrows()):
            with col:
                render_recommendation_card(rec, f"{key_prefix}_{idx}")


def show_empty_state(title, message):
    st.markdown(
        f"""
        <div class="empty-state">
            <strong>{escape(title)}</strong>
            <span>{escape(message)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="CMA Art Recommender",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(STYLES, unsafe_allow_html=True)

    df, embeddings = load_data()
    model = load_model()

    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "Browse by artwork"

    render_top_shell(len(df))

    active_tab = st.radio(
        "Choose a discovery method",
        ["Browse by artwork", "Describe a painting", "Visual Search"],
        horizontal=True,
        key="app_mode",
        label_visibility="collapsed",
    )

    st.write("")

    if active_tab == "Browse by artwork":
        render_section_heading(
            "Artwork to artwork",
            "Start from a painting you already know",
            "Search by title or artist, filter the collection.",
        )

        with st.container(border=True):
            c1, c2, c3 = st.columns([2.2, 2.2, 0.95], gap="medium")
            with c1:
                search_query = st.text_input(
                    "Search collection",
                    placeholder="Search by title or artist",
                    key="search_q",
                )
            with c2:
                departments = sorted(df["department"].dropna().unique().tolist())
                selected_departments = st.multiselect(
                    "Department",
                    departments,
                    default=[],
                    placeholder="All departments",
                    key="department_filter",
                )
            with c3:
                k = st.number_input(
                    "Recommendations",
                    min_value=3,
                    max_value=20,
                    value=10,
                    step=1,
                    key="similar_result_count",
                )

        filtered_df = df[df["department"].isin(selected_departments)] if selected_departments else df

        if search_query:
            mask = (
                filtered_df["title"].str.contains(search_query, case=False, na=False)
                | filtered_df["artist"].str.contains(search_query, case=False, na=False)
            )
            search_results = filtered_df[mask]
        else:
            search_results = filtered_df

        st.write("")

        if search_results.empty:
            show_empty_state(
                "No artworks found",
                "Try a shorter title, a different artist spelling, or remove a department filter.",
            )
            return

        display_labels = [
            f"{value_or(row.get('title'), 'Untitled')} — {value_or(row.get('artist'), 'Unknown artist')}"
            for _, row in search_results.iterrows()
        ]

        with st.container(border=True):
            selected_label = st.selectbox("Selected artwork", display_labels, key="artwork_selector")

        selected_row = search_results.iloc[display_labels.index(selected_label)]

        if "pinned_idx" in st.session_state:
            pinned_idx = st.session_state.pop("pinned_idx")
            pinned_rows = df[df["embedding_index"] == pinned_idx]
            if not pinned_rows.empty:
                selected_row = pinned_rows.iloc[0]

        query_idx = int(selected_row["embedding_index"])
        recs = recommend(query_idx, embeddings, df, k=k)

        st.write("")
        left, right = st.columns([1.2, 1.8], gap="large")

        with left:
            with st.container(border=True):
                render_artwork_details(selected_row, label="Selected artwork")

        with right:
            render_section_heading(
                "Recommendations",
                f"{len(recs)} paintings with the closest match.",
                "Visually and conceptually related artworks from the collection.",
            )
            st.markdown(
                '<div class="recs-header"><span class="recs-chip">Large card gallery</span><span class="recs-chip">Similarity-based results</span></div>',
                unsafe_allow_html=True,
            )
            show_recommendations_grid(recs, key_prefix="similar")

    elif active_tab == "Describe a painting":
        render_section_heading(
            "Text to artwork",
            "Describe the painting you want to discover",
            "Use subject, mood, era, palette, or composition.",
        )

        with st.container(border=True):
            text_query = st.text_input(
                "Your description",
                placeholder="e.g. calm Japanese landscape · dramatic stormy sea · portrait of a woman in red",
                key="text_query_input",
            )

            st.markdown('<div class="section-label" style="margin-top:12px;">Try an example</div>', unsafe_allow_html=True)
            examples = [
                "calm Japanese landscape",
                "dramatic stormy sea",
                "portrait 19th century France",
                "abstract geometric modern",
            ]
            example_cols = st.columns(4, gap="small")
            for col, example in zip(example_cols, examples):
                with col:
                    st.button(
                        example,
                        key=f"example_{example}",
                        type="secondary",
                        use_container_width=True,
                        on_click=set_example_query,
                        args=(example,),
                    )

            c1, c2, c3 = st.columns([1, 1, 1.8], gap="medium")
            with c1:
                k = st.number_input(
                    "Recommendations",
                    min_value=3,
                    max_value=20,
                    value=10,
                    step=1,
                    key="text_result_count",
                )
            with c2:
                # This invisible spacer matches the exact height of the input label
                st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
                search_clicked = st.button(
                    "Find paintings",
                    key="find_paintings",
                    type="primary",
                    use_container_width=True,
                )
            with c3:
                st.markdown(
                    '<p class="section-copy" style="padding-top:36px;">Results stay visible while you refine the text query.</p>',
                    unsafe_allow_html=True,
                )

        if search_clicked and text_query.strip():
            st.session_state["text_query_vec"] = model.encode(text_query.strip(), normalize_embeddings=True)
            st.session_state["text_query_label"] = text_query.strip()
            st.session_state["text_query"] = text_query.strip()

        st.write("")

        if "text_query_vec" not in st.session_state:
            show_empty_state(
                "What are you looking for?",
                "Describe a painting above, then click “Find paintings” to see the closest matches.",
            )
            return

        recs = recommend_by_vector(st.session_state["text_query_vec"], embeddings, df, k=k)
        query_label = value_or(st.session_state.get("text_query_label"), "Your description")

        if recs.empty:
            show_empty_state(
                "No matches returned",
                "Try a simpler or more descriptive phrase and search again.",
            )
            return

        st.markdown(
            f'<div class="query-callout"><strong>Searching for:</strong> {escape(query_label)}</div>',
            unsafe_allow_html=True,
        )

        render_section_heading(
            "Matches",
            f"{len(recs)} closest paintings for this description",
            "Best match first — click any painting to explore similar works.",
        )
        st.markdown(
            '<div class="recs-header"><span class="recs-chip">Semantic text search</span><span class="recs-chip">Best match first</span></div>',
            unsafe_allow_html=True,
        )
        show_recommendations_grid(recs, key_prefix="text")

    else:
        # ── Tab 3: Visual Search (CLIP) ─────────────────────────────────────
        render_section_heading(
            "Image to artwork",
            "Search by visual appearance with CLIP",
            "Describe what a painting looks like, or upload your own image — "
            "CLIP searches the collection.",
        )

        clip_embeddings = load_clip_embeddings()

        if clip_embeddings is None:
            st.info(
                "CLIP embeddings not yet generated. "
                "Run `python src/4_embed_images_clip.py` (or submit job_clip_embed.yaml) first."
            )
            return

        # Merge clip_success into a working copy of df so recommend functions can filter
        clip_index = load_clip_index()
        if clip_index is not None:
            df_clip = df.copy()
            df_clip["clip_success"] = clip_index["clip_success"].values
        else:
            df_clip = df

        clip_model, clip_processor = load_clip_model()

        sub_mode = st.radio(
            "Input type",
            ["Describe visually", "Upload an image"],
            horizontal=True,
            key="clip_sub_mode",
            label_visibility="collapsed",
        )

        st.write("")

        with st.container(border=True):
            if sub_mode == "Describe visually":
                clip_query = st.text_input(
                    "Visual description",
                    placeholder="e.g. red dramatic painting with horses · misty blue landscape · gold and black portrait",
                    key="clip_text_input",
                )
                
                # Swapped order and updated columns to [1, 1] for even widths
                c1, c2 = st.columns([1, 1], gap="medium")
                with c1:
                    clip_k = st.number_input(
                        "Recommendations",
                        min_value=3,
                        max_value=20,
                        value=10,
                        step=1,
                        key="clip_result_count",
                    )
                with c2:
                    # Invisible spacer to push the button down to align with the input
                    st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
                    clip_search_clicked = st.button(
                        "Find paintings",
                        key="clip_find",
                        type="primary",
                        use_container_width=True,
                    )

                if clip_search_clicked and clip_query.strip():
                    st.session_state["clip_query_vec"] = recommend_by_clip_text(
                        clip_query.strip(), clip_embeddings, df_clip,
                        clip_model, clip_processor, k=int(clip_k),
                    )
                    st.session_state["clip_query_label"] = clip_query.strip()

            else:
                uploaded = st.file_uploader(
                    "Upload a painting or photo",
                    type=["jpg", "jpeg", "png"],
                    key="clip_upload",
                )
                clip_k = st.number_input(
                    "Recommendations",
                    min_value=3,
                    max_value=20,
                    value=10,
                    step=1,
                    key="clip_result_count_img",
                )

                if uploaded is not None:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(uploaded).convert("RGB")
                    st.image(pil_img, caption="Your uploaded image", width=280)
                    recs_clip = recommend_by_clip_image(
                        pil_img, clip_embeddings, df_clip,
                        clip_model, clip_processor, k=int(clip_k),
                    )
                    st.session_state["clip_query_vec"] = recs_clip
                    st.session_state["clip_query_label"] = f"Image upload: {uploaded.name}"

        st.write("")

        if "clip_query_vec" not in st.session_state:
            show_empty_state(
                "Ready to search visually",
                "Describe what you're looking for, or upload an image above.",
            )
            return

        recs_clip = st.session_state["clip_query_vec"]
        clip_label = value_or(st.session_state.get("clip_query_label"), "Visual query")

        if isinstance(recs_clip, type(None)) or (hasattr(recs_clip, "empty") and recs_clip.empty):
            show_empty_state("No matches returned", "Try a different description or image.")
            return

        st.markdown(
            f'<div class="query-callout"><strong>Searching for:</strong> {escape(clip_label)}</div>',
            unsafe_allow_html=True,
        )
        render_section_heading(
            f"Top {len(recs_clip)} visual matches · CLIP ViT-B/32",
            "Closest paintings by visual appearance",
            "Results come from comparing pixel-level image features, not text descriptions.",
        )
        st.markdown(
            '<div class="recs-header">'
            '<span class="recs-chip">CLIP ViT-B/32</span>'
            '<span class="recs-chip">Visual similarity</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        show_recommendations_grid(recs_clip, key_prefix="clip")


if __name__ == "__main__":
    main()
