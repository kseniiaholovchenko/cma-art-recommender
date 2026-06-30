"""
Parse raw Label Studio export into a clean CSV.
Input:  <project_root>/src/annotation/annotation_export.json  (exported from Label Studio)
Output: <project_root>/src/annotation/annotations_clean.csv

Expected Label Studio result format:
  result[n].from_name in {"mood", "subject", "style_period"}
  result[n].value.choices[0]  →  the selected label
"""

import os
import json
import pandas as pd

ANNOTATION_DIR = os.path.dirname(__file__)


def parse_task(task, image_to_id):
    """
    Extract id, mood, subject, and style_period from one Label Studio task.
    Parameters:
        task (dict): One task dict from the Label Studio JSON export.
        image_to_id (dict): image_url → artwork id, built from tasks.json.
    Returns:
        dict: {id, mood, subject, style_period} or None if unannotated.
    """
    data = task.get("data", {})
    # data["id"] exists in new-format tasks; fall back to image URL lookup for existing export
    task_id = data.get("id") or image_to_id.get(data.get("image"))

    if task_id is None:
        return None

    annotations = task.get("annotations", [])

    if not annotations:
        return None

    # Use the last annotation — Label Studio exports oldest-first, so [-1] is most recent
    results = annotations[-1].get("result", [])

    mood = None
    subject = None
    style_period = None

    for entry in results:
        from_name = entry.get("from_name")
        choices = entry.get("value", {}).get("choices", [])
        value = choices[0] if choices else None

        if from_name == "mood":
            mood = value
        elif from_name == "subject":
            subject = value
        elif from_name == "style_period":
            style_period = value

    if mood is None and subject is None and style_period is None:
        return None

    return {"id": task_id, "mood": mood, "subject": subject, "style_period": style_period}


if __name__ == "__main__":
    # Build image URL → artwork id lookup from tasks.json to recover original CMA IDs
    tasks_path = os.path.join(ANNOTATION_DIR, "tasks.json")
    with open(tasks_path, encoding="utf-8") as f:
        task_defs = json.load(f)
    image_to_id = {
        t["data"]["image"]: t.get("data", {}).get("id") or t.get("id")
        for t in task_defs
        if t.get("data", {}).get("image")
    }

    export_path = os.path.join(ANNOTATION_DIR, "annotation_export.json")
    print(f"Loading {export_path}...")
    with open(export_path, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"  Loaded {len(tasks)} tasks.")

    rows = [parse_task(t, image_to_id) for t in tasks]
    rows = [r for r in rows if r is not None]

    df = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} annotated tasks.")

    for col in ["mood", "subject", "style_period"]:
        if col in df.columns:
            print(f"\n{col} value counts:")
            print(df[col].value_counts().to_string())

    output_path = os.path.join(ANNOTATION_DIR, "annotations_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
