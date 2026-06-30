"""
Evaluate recommender quality with Precision@k and Recall@k across 8 methods
(MiniLM/CLIP embeddings x same-artist/mood/subject ground truth, plus a
medium-only baseline and a random baseline), then log both the evaluation
results and the embedding/experiment configs to Weights & Biases.

Input:  ../data/artworks_with_index.csv   (contains mood/subject columns for annotated rows)
        ../data/embeddings.npy            (MiniLM text embeddings)
        ../data/image_embeddings_clip.npy (CLIP image embeddings, optional)
        ../data/clip_index.csv            (CLIP success mask, optional)
        ../data/quality_report.json       (from 2_data_quality.py)
Output: Comparison table printed to console.
        Two W&B runs in project "cleveland-art-recommender":
          run-minilm-l6   — MiniLM config, embedding stats, key eval metrics,
                             and the full 8-method comparison table.
          run-clip-vitb32 — CLIP config, embedding stats, key eval metrics
                             (skipped if image_embeddings_clip.npy is absent).

All methods use the same full search space (1154 artworks) so metrics are
directly comparable. Method 4 (random baseline) shows the evaluation floor
and how much the real model beats pure chance.
"""

import json
import os
import random
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))


def load_quality_metrics():
    """Load the quality report produced by 2_data_quality.py."""
    report_path = os.path.join(DATA_DIR, "quality_report.json")
    with open(report_path) as f:
        return json.load(f)


def precision_at_k(recommended_ids, relevant_ids, k):
    """
    Fraction of the top-k recommended items that are relevant.
    Parameters:
        recommended_ids (list): Ordered list of recommended row indices.
        relevant_ids (set): Set of relevant row indices for this query.
        k (int): Cut-off rank.
    Returns:
        float: Precision@k in [0, 1].
    """
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k):
    """
    Fraction of all relevant items that appear in the top-k recommendations.
    Parameters:
        recommended_ids (list): Ordered list of recommended row indices.
        relevant_ids (set): Set of relevant row indices for this query.
        k (int): Cut-off rank.
    Returns:
        float: Recall@k in [0, 1]. Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def get_top_n(query_idx, embeddings, n=10):
    """
    Return the top-n most similar row indices (excluding the query itself).
    Parameters:
        query_idx (int): Index of the query artwork.
        embeddings (np.ndarray): Normalized embedding matrix.
        n (int): Number of neighbors to return.

    Returns:
        list of int: Top-n row indices in descending similarity order.
    """
    query_vec = embeddings[query_idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]
    sims[query_idx] = -1  # exclude self
    return list(np.argsort(sims)[::-1][:n])


def evaluate_random_baseline(df, artist_to_indices, eligible_indices, k_values, sample_size):
    """
    Random baseline: recommend k artworks chosen uniformly at random.
    Accepts pre-computed artist_to_indices and eligible_indices (same values
    used by evaluate_method) so the dict is not rebuilt redundantly.

    Expected P@k ≈ relevant_count / n_artworks — shows how much the real
    model improves over pure chance.

    Parameters:
        df (pd.DataFrame): Artworks dataframe (reset_index applied).
        artist_to_indices (dict): artist → list of df index labels.
        eligible_indices (list): df index labels of eligible query artworks.
        k_values (list of int): k cut-offs to evaluate.
        sample_size (int): How many query artworks to sample.

    Returns:
        dict: Mean metrics {P@5, R@5, P@10, R@10}.
    """
    if not eligible_indices:
        return {}

    all_indices = list(df.index)
    sample = random.sample(eligible_indices, min(sample_size, len(eligible_indices)))

    metrics = {f"P@{k}": [] for k in k_values}
    metrics.update({f"R@{k}": [] for k in k_values})

    max_k = max(k_values)
    for query_idx in sample:
        artist = df.loc[query_idx, "artist"]
        relevant = set(artist_to_indices[artist]) - {query_idx}
        # Sample one extra so we always get max_k results even if query_idx is drawn
        pool = random.sample(all_indices, max_k + 1)
        rec_indices = [i for i in pool if i != query_idx][:max_k]
        for k in k_values:
            metrics[f"P@{k}"].append(precision_at_k(rec_indices, relevant, k))
            metrics[f"R@{k}"].append(recall_at_k(rec_indices, relevant, k))

    return {metric: round(float(np.mean(vals)), 4) for metric, vals in metrics.items()}


def evaluate_method(embeddings_matrix, df, artist_to_indices, eligible_indices, k_values, sample_size):
    """
    Evaluate a recommender using the same-artist proxy for ground truth.
    Accepts pre-computed artist_to_indices and eligible_indices so the dict
    is not rebuilt for every method call.
    Parameters:
        embeddings_matrix (np.ndarray): Embedding matrix aligned with df rows.
        df (pd.DataFrame): Artworks dataframe with an "artist" column.
        artist_to_indices (dict): artist → list of df index labels.
        eligible_indices (list): df index labels of eligible query artworks.
        k_values (list of int): k cut-offs to evaluate, e.g. [5, 10].
        sample_size (int): How many query artworks to sample.

    Returns:
        dict: Mean metrics {P@5: float, R@5: float, P@10: float, R@10: float}.
    """
    if not eligible_indices:
        print("  No eligible artists with >= 3 artworks — skipping.")
        return {}

    sample = random.sample(eligible_indices, min(sample_size, len(eligible_indices)))

    metrics = {f"P@{k}": [] for k in k_values}
    metrics.update({f"R@{k}": [] for k in k_values})

    max_k = max(k_values)
    for query_idx in sample:
        artist = df.loc[query_idx, "artist"]
        relevant = set(artist_to_indices[artist]) - {query_idx}
        rec_indices = get_top_n(query_idx, embeddings_matrix, n=max_k)

        for k in k_values:
            metrics[f"P@{k}"].append(precision_at_k(rec_indices, relevant, k))
            metrics[f"R@{k}"].append(recall_at_k(rec_indices, relevant, k))

    return {metric: round(float(np.mean(vals)), 4) for metric, vals in metrics.items()}


if __name__ == "__main__":
    index_path = os.path.join(DATA_DIR, "artworks_with_index.csv")
    embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")

    print(f"Loading {index_path}...")
    # reset_index so df.loc[i] aligns with embeddings[i]
    df = pd.read_csv(index_path).reset_index(drop=True)
    print(f"Loading {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"  {len(df)} artworks, shape: {embeddings.shape}")

    k_values = [5, 10]
    sample_size = 200

    # Compute once — reused by Methods 1, 4, and 2 so the groupby runs only once.
    artist_to_indices = (
        df.groupby("artist").apply(lambda g: list(g.index), include_groups=False).to_dict()
    )
    eligible_indices = [
        i for i, row in df.iterrows()
        if len(artist_to_indices.get(row["artist"], [])) >= 3
    ]
    print(f"  {len(eligible_indices)} artworks eligible (artist with >= 3 paintings).")

    results = {}

    # Each method resets to seed 42 before sampling query artworks.
    # Methods 1, 4, and 2 all draw from the same eligible_indices pool, so
    # resetting to the same seed makes them evaluate on the identical 200 queries —
    # directly comparable, reproducible, and consistent across W&B runs.
    random.seed(42)
    print("\n=== Method 1: MiniLM + same-artist proxy ===")
    m1 = evaluate_method(embeddings, df, artist_to_indices, eligible_indices, k_values, sample_size)
    results["method1_same_artist"] = m1
    print(f"  {m1}")

    # Method 4: Random baseline — same ground truth as Method 1, pure chance
    # Shows the evaluation floor and how much Method 1 actually beats chance.
    random.seed(42)
    print("\n=== Method 4: Random baseline ===")
    m4 = evaluate_random_baseline(df, artist_to_indices, eligible_indices, k_values, sample_size)
    results["method4_random"] = m4
    print(f"  {m4}")

    # Method 2: Medium-only baseline
    # Re-encode using only the medium field (e.g. "oil on canvas") to show
    # that full-text embedding outperforms a naive single-field approach.
    random.seed(42)
    print("\n=== Method 2: Medium-only baseline ===")
    print("  Re-encoding using medium field only...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    medium_texts = df["medium"].fillna("").tolist()
    medium_embeddings = model.encode(
        medium_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    m2 = evaluate_method(medium_embeddings, df, artist_to_indices, eligible_indices, k_values, sample_size)
    results["method2_medium_only"] = m2
    print(f"  {m2}")

    # Method 3: MiniLM + mood (same mood label) — uses full embeddings
    # artworks_with_index.csv already has the mood column (NaN for unannotated rows)
    # so no separate file load is needed.
    # Crucially, we search all 1154 embeddings — same space as Methods 1 and 2 —
    # so the metrics are directly comparable.
    random.seed(42)
    df_annotated = df[df["mood"].notna()]
    print(f"\n=== Method 3: MiniLM + mood (same mood label) ===")
    print(f"  {len(df_annotated)} annotated artworks used as queries; "
          f"search space: all {len(df)} artworks.")

    if len(df_annotated) < 10:
        print("  Too few annotated artworks — skipping Method 3.")
    else:
        # mood_to_indices: mood → list of df index labels (= embedding row indices)
        # Note: we do NOT reset_index on df_annotated, so g.index gives the
        # original df labels (0..1153), which equal embedding row indices.
        mood_to_indices = (
            df_annotated.groupby("mood")
            .apply(lambda g: list(g.index), include_groups=False)
            .to_dict()
        )
        eligible_mood = [
            i for i in df_annotated.index
            if len(mood_to_indices.get(df.loc[i, "mood"], [])) >= 3
        ]
        if eligible_mood:
            sample_mood = random.sample(eligible_mood, min(sample_size, len(eligible_mood)))
            m3_metrics = {f"P@{k}": [] for k in k_values}
            m3_metrics.update({f"R@{k}": [] for k in k_values})
            max_k = max(k_values)
            for qi in sample_mood:
                mood = df.loc[qi, "mood"]
                relevant = set(mood_to_indices[mood]) - {qi}
                recs = get_top_n(qi, embeddings, n=max_k)
                for k in k_values:
                    m3_metrics[f"P@{k}"].append(precision_at_k(recs, relevant, k))
                    m3_metrics[f"R@{k}"].append(recall_at_k(recs, relevant, k))
            m3 = {met: round(float(np.mean(v)), 4) for met, v in m3_metrics.items()}
            results["method3_annotations"] = m3
            print(f"  {m3}")
        else:
            print("  No mood category has >= 3 artworks — skipping Method 3.")

    # Method 7: MiniLM + subject (same subject label) — uses full embeddings
    # Same pattern as Method 3, but ground truth is the subject column
    # (e.g. "portrait", "landscape", "still life") instead of mood.
    random.seed(42)
    df_annotated_subject = df[df["subject"].notna()]
    print(f"\n=== Method 7: MiniLM + subject (same subject label) ===")
    print(f"  {len(df_annotated_subject)} annotated artworks used as queries; "
          f"search space: all {len(df)} artworks.")

    if len(df_annotated_subject) < 10:
        print("  Too few annotated artworks — skipping Method 7.")
    else:
        subject_to_indices = (
            df_annotated_subject.groupby("subject")
            .apply(lambda g: list(g.index), include_groups=False)
            .to_dict()
        )
        eligible_subject = [
            i for i in df_annotated_subject.index
            if len(subject_to_indices.get(df.loc[i, "subject"], [])) >= 3
        ]
        if eligible_subject:
            sample_subject = random.sample(
                eligible_subject, min(sample_size, len(eligible_subject))
            )
            m7_metrics = {f"P@{k}": [] for k in k_values}
            m7_metrics.update({f"R@{k}": [] for k in k_values})
            max_k = max(k_values)
            for qi in sample_subject:
                subject = df.loc[qi, "subject"]
                relevant = set(subject_to_indices[subject]) - {qi}
                recs = get_top_n(qi, embeddings, n=max_k)
                for k in k_values:
                    m7_metrics[f"P@{k}"].append(precision_at_k(recs, relevant, k))
                    m7_metrics[f"R@{k}"].append(recall_at_k(recs, relevant, k))
            m7 = {met: round(float(np.mean(v)), 4) for met, v in m7_metrics.items()}
            results["method7_minilm_subject"] = m7
            print(f"  {m7}")
        else:
            print("  No subject category has >= 3 artworks — skipping Method 7.")

    # Method 5: CLIP + same-artist proxy (visual vectors)
    clip_path = os.path.join(DATA_DIR, "image_embeddings_clip.npy")
    print(f"\n=== Method 5: CLIP + same-artist proxy ===")
    if not os.path.exists(clip_path):
        print("  image_embeddings_clip.npy not found — skipping Method 5. "
              "Run 4_embed_images_clip.py first.")
    else:
        clip_embeddings = np.load(clip_path)
        # Only query artworks where CLIP succeeded (zero vectors skew metrics)
        clip_index_path = os.path.join(DATA_DIR, "clip_index.csv")
        if os.path.exists(clip_index_path):
            clip_idx = pd.read_csv(clip_index_path)
            success_set = set(clip_idx.index[clip_idx["clip_success"]])
            eligible_clip = [i for i in eligible_indices if i in success_set]
        else:
            eligible_clip = eligible_indices
        print(f"  {len(eligible_clip)} eligible query artworks with successful CLIP encoding.")
        random.seed(42)
        m5 = evaluate_method(
            clip_embeddings, df, artist_to_indices, eligible_clip, k_values, sample_size
        )
        results["method5_clip_image"] = m5
        print(f"  {m5}")

        # Method 6: CLIP + mood ground truth (visual mood detection)
        # Same mood-based proxy as Method 3, but evaluated against CLIP image
        # embeddings — tests whether visual similarity alone recovers the
        # mood groupings a human annotator assigned.
        print(f"\n=== Method 6: CLIP + mood ground truth (visual mood detection) ===")
        df_annotated_clip = df[df["mood"].notna()]
        if os.path.exists(clip_index_path):
            df_annotated_clip = df_annotated_clip[df_annotated_clip.index.isin(success_set)]
        print(f"  {len(df_annotated_clip)} annotated artworks with successful CLIP encoding "
              f"used as queries; search space: all {len(df)} artworks.")

        if len(df_annotated_clip) < 10:
            print("  Too few annotated+CLIP-encoded artworks — skipping Method 6.")
        else:
            mood_to_indices_clip = (
                df_annotated_clip.groupby("mood")
                .apply(lambda g: list(g.index), include_groups=False)
                .to_dict()
            )
            clip_eligible_mood = [
                i for i in df_annotated_clip.index
                if len(mood_to_indices_clip.get(df.loc[i, "mood"], [])) >= 3
            ]
            if clip_eligible_mood:
                random.seed(42)
                sample_mood_clip = random.sample(
                    clip_eligible_mood, min(sample_size, len(clip_eligible_mood))
                )
                m6_metrics = {f"P@{k}": [] for k in k_values}
                m6_metrics.update({f"R@{k}": [] for k in k_values})
                max_k = max(k_values)
                for qi in sample_mood_clip:
                    mood = df.loc[qi, "mood"]
                    relevant = set(mood_to_indices_clip[mood]) - {qi}
                    recs = get_top_n(qi, clip_embeddings, n=max_k)
                    for k in k_values:
                        m6_metrics[f"P@{k}"].append(precision_at_k(recs, relevant, k))
                        m6_metrics[f"R@{k}"].append(recall_at_k(recs, relevant, k))
                m6 = {met: round(float(np.mean(v)), 4) for met, v in m6_metrics.items()}
                results["method6_clip_mood"] = m6
                print(f"  {m6}")
            else:
                print("  No mood category has >= 3 CLIP-encoded artworks — skipping Method 6.")

        # Method 8: CLIP + subject ground truth
        # Same pattern as Method 6, but ground truth is the subject column
        # instead of mood — tests whether visual similarity recovers subject
        # categories (e.g. portrait, landscape, still life).
        print(f"\n=== Method 8: CLIP + subject ground truth ===")
        df_annotated_clip_subject = df[df["subject"].notna()]
        if os.path.exists(clip_index_path):
            df_annotated_clip_subject = df_annotated_clip_subject[
                df_annotated_clip_subject.index.isin(success_set)
            ]
        print(f"  {len(df_annotated_clip_subject)} annotated artworks with successful CLIP "
              f"encoding used as queries; search space: all {len(df)} artworks.")

        if len(df_annotated_clip_subject) < 10:
            print("  Too few annotated+CLIP-encoded artworks — skipping Method 8.")
        else:
            subject_to_indices_clip = (
                df_annotated_clip_subject.groupby("subject")
                .apply(lambda g: list(g.index), include_groups=False)
                .to_dict()
            )
            clip_eligible_subject = [
                i for i in df_annotated_clip_subject.index
                if len(subject_to_indices_clip.get(df.loc[i, "subject"], [])) >= 3
            ]
            if clip_eligible_subject:
                random.seed(42)
                sample_subject_clip = random.sample(
                    clip_eligible_subject, min(sample_size, len(clip_eligible_subject))
                )
                m8_metrics = {f"P@{k}": [] for k in k_values}
                m8_metrics.update({f"R@{k}": [] for k in k_values})
                max_k = max(k_values)
                for qi in sample_subject_clip:
                    subject = df.loc[qi, "subject"]
                    relevant = set(subject_to_indices_clip[subject]) - {qi}
                    recs = get_top_n(qi, clip_embeddings, n=max_k)
                    for k in k_values:
                        m8_metrics[f"P@{k}"].append(precision_at_k(recs, relevant, k))
                        m8_metrics[f"R@{k}"].append(recall_at_k(recs, relevant, k))
                m8 = {met: round(float(np.mean(v)), 4) for met, v in m8_metrics.items()}
                results["method8_clip_subject"] = m8
                print(f"  {m8}")
            else:
                print("  No subject category has >= 3 CLIP-encoded artworks — skipping Method 8.")

    # --- Print comparison table ---
    print("\n=== Comparison table ===")
    method_labels = {
        "method1_same_artist": "Method 1 (MiniLM + same-artist)",
        "method4_random":      "Method 4 (random baseline)",
        "method2_medium_only": "Method 2 (medium-only)",
        "method3_annotations": "Method 3 (MiniLM + mood)",
        "method5_clip_image":  "Method 5 (CLIP + same-artist)",
        "method6_clip_mood":   "Method 6 (CLIP + mood)",
        "method7_minilm_subject": "Method 7 (MiniLM + subject)",
        "method8_clip_subject":   "Method 8 (CLIP + subject)",
    }
    print(f"  {'Method':<30} {'P@5':>8} {'R@5':>8} {'P@10':>8} {'R@10':>8}")
    print("  " + "-" * 66)
    for key, label in method_labels.items():
        if key not in results:
            continue
        m = results[key]
        print(
            f"  {label:<30} {str(m.get('P@5', 'N/A')):>8} {str(m.get('R@5', 'N/A')):>8}"
            f" {str(m.get('P@10', 'N/A')):>8} {str(m.get('R@10', 'N/A')):>8}"
        )

    # --- Log to W&B ---
    print("\nLogging to W&B project 'cleveland-art-recommender'...")

    def build_comparison_table(rows):
        """Build a wandb.Table from a list of (label, metrics dict) rows."""
        table = wandb.Table(columns=["Method", "P@5", "R@5", "P@10", "R@10"])
        for label, m in rows:
            if m:
                table.add_data(
                    label,
                    m.get("P@5"), m.get("R@5"),
                    m.get("P@10"), m.get("R@10"),
                )
        return table

    quality_metrics = load_quality_metrics()

    # Run 1: MiniLM — embedding config + only the MiniLM-based methods
    # (1, 2, 3, 7), plus the random baseline for reference. Logged as a single
    # table (not individual wandb.log scalars) so it renders as a clean
    # comparison instead of a scatter of single-point charts.
    minilm_rows = [
        ("Method 1: MiniLM + same-artist", results.get("method1_same_artist", {})),
        ("Method 2: Medium-only (MiniLM)", results.get("method2_medium_only", {})),
        ("Method 3: MiniLM + mood",        results.get("method3_annotations", {})),
        ("Method 7: MiniLM + subject",     results.get("method7_minilm_subject", {})),
        ("Method 4: Random baseline",      results.get("method4_random", {})),
    ]
    with wandb.init(
        project="cleveland-art-recommender",
        name="run-minilm-l6",
        config={
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "modality": "text",
            "normalize": True,
            "text_strategy": "title+artist+medium+culture+type+description+did_you_know+mood+subject+style_period",
            "k_neighbors": 10,
        },
    ):
        wandb.log({
            "dataset_size": quality_metrics.get("total_after_cleaning"),
            "embedding_shape_rows": embeddings.shape[0],
            "embedding_shape_cols": embeddings.shape[1],
            "mean_vector_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        })
        wandb.log({"minilm_methods": build_comparison_table(minilm_rows)})
        print("  Logged run-minilm-l6.")

    # Run 2: CLIP — embedding config + only the CLIP-based methods
    # (5, 6, 8), plus the random baseline for reference.
    # Only runs if image_embeddings_clip.npy was found (Method 5 populated results).
    if "method5_clip_image" in results:
        clip_rows = [
            ("Method 5: CLIP + same-artist", results.get("method5_clip_image", {})),
            ("Method 6: CLIP + mood",        results.get("method6_clip_mood", {})),
            ("Method 8: CLIP + subject",     results.get("method8_clip_subject", {})),
            ("Method 4: Random baseline",    results.get("method4_random", {})),
        ]
        with wandb.init(
            project="cleveland-art-recommender",
            name="run-clip-vitb32",
            config={
                "embedding_model": "openai/clip-vit-base-patch32",
                "embedding_dim": 512,
                "modality": "image",
                "normalize": True,
                "text_strategy": "image pixels (no text)",
                "k_neighbors": 10,
            },
        ):
            wandb.log({
                "dataset_size": quality_metrics.get("total_after_cleaning"),
                "embedding_shape_rows": clip_embeddings.shape[0],
                "embedding_shape_cols": clip_embeddings.shape[1],
                "mean_vector_norm": float(np.linalg.norm(clip_embeddings, axis=1).mean()),
            })
            wandb.log({"clip_methods": build_comparison_table(clip_rows)})
            print("  Logged run-clip-vitb32.")
    else:
        print("  Skipping run-clip-vitb32 — CLIP embeddings not available.")

    # Run 3: Full comparison — all 8 methods side by side in one place.
    method_rows = [
        ("MiniLM + same-artist", results.get("method1_same_artist", {})),
        ("Random baseline",      results.get("method4_random", {})),
        ("Medium only",          results.get("method2_medium_only", {})),
        ("Annotations",          results.get("method3_annotations", {})),
        ("CLIP + same-artist",   results.get("method5_clip_image", {})),
        ("CLIP + mood",          results.get("method6_clip_mood", {})),
        ("MiniLM + subject",     results.get("method7_minilm_subject", {})),
        ("CLIP + subject",       results.get("method8_clip_subject", {})),
    ]
    with wandb.init(
        project="cleveland-art-recommender",
        name="run-full-comparison",
        config={"k_values": k_values, "sample_size": sample_size},
    ):
        wandb.log({"evaluation_comparison": build_comparison_table(method_rows)})
        print("  Logged run-full-comparison.")

    print("\nDone.")
