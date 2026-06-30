"""
Precompute embedding vectors for all dev tasks.

Usage:
    python scripts/precompute_vectors.py

Output:
    src/gensie/extraction/data/dev_vectors.json  — vectors + metadata
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastembed import TextEmbedding


DEV_DIR = _PROJECT_ROOT / "data" / "dev"
OUTPUT_DIR = _PROJECT_ROOT / "src" / "gensie" / "extraction" / "data"
OUTPUT_PATH = OUTPUT_DIR / "dev_vectors.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_fingerprint(task: dict) -> str:
    """Build text fingerprint for a task (same as SchemaAnalyzer._build_fingerprint)."""
    schema = task.get("target_schema", {})
    title = schema.get("title", "")
    raw_desc = schema.get("description", "")
    desc = raw_desc.split("\n")[0].strip()
    props = schema.get("properties", {})
    field_names = sorted(props.keys())
    field_str = ", ".join(field_names)
    instruction = task.get("instruction", "")[:250]
    meta = task.get("metadata", {})
    domain = meta.get("domain", "")
    subdomain = meta.get("subdomain", "")
    domain_str = f"{domain} ({subdomain})" if subdomain else domain

    return (
        f"Schema: {title} | Description: {desc}"
        f" | Domain: {domain_str} | Fields: {field_str}"
        f" | Task: {instruction}"
    )


def extract_complexity(task: dict) -> tuple[str, str]:
    """Extract complexity level from schema description."""
    desc = task.get("target_schema", {}).get("description", "")
    for line in desc.split("\n"):
        line = line.strip()
        if "Complexity:" in line:
            # "Complexity: L5 (Hierarchical Grouping)."
            parts = line.replace("Complexity:", "").strip()
            level = parts.split()[0].strip()  # "L5"
            # Get description: everything between ( and )
            desc_part = parts[parts.index("(") + 1 : parts.index(")")] if "(" in parts else ""
            return level, desc_part
    return "L4", "General"


def extract_domain(task: dict) -> str:
    """Extract domain from metadata or schema title."""
    meta = task.get("metadata", {})
    domain = meta.get("domain", "")
    if domain:
        return domain.lower()
    # Fallback: infer from task id prefix
    task_id = task.get("id", "")
    if "_" in task_id:
        return task_id.split("_")[0].lower()
    return "general"


def extract_field_types(task: dict) -> list[str]:
    """Detect types present in schema (same as SchemaAnalyzer._detect_field_types)."""
    props = task.get("target_schema", {}).get("properties", {})
    types = set()
    for name, prop in props.items():
        if not isinstance(prop, dict):
            continue
        if "enum" in prop:
            types.add("enum")
        raw_type = prop.get("type")
        if isinstance(raw_type, str):
            types.add(raw_type)
        elif isinstance(raw_type, list):
            non_null = [t for t in raw_type if t != "null"]
            if non_null:
                types.add(non_null[0])
        if raw_type == "array" or (isinstance(raw_type, list) and "array" in raw_type):
            items = prop.get("items", {})
            if isinstance(items, dict) and items.get("type") == "object":
                types.add("nested_object")
        if raw_type == "object":
            types.add("nested_object")
    priority = ["array", "enum", "nested_object", "number", "integer", "boolean", "string"]
    ordered = [t for t in priority if t in types]
    for t in sorted(types):
        if t not in ordered:
            ordered.append(t)
    return ordered


def main():
    print(f"Loading dev tasks from {DEV_DIR}")
    dev_files = sorted(Path(DEV_DIR).glob("*.json"))
    print(f"Found {len(dev_files)} tasks")

    tasks = []
    for f in dev_files:
        with open(f, encoding="utf-8") as fp:
            task = json.load(fp)
        tasks.append(task)

    print("Building fingerprints...")
    fingerprints = [build_fingerprint(t) for t in tasks]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = TextEmbedding(MODEL_NAME)

    print("Computing embeddings...")
    vectors = [emb.tolist() for emb in model.embed(fingerprints)]
    print(f"Computed {len(vectors)} vectors, dim={len(vectors[0])}")

    # Build metadata
    metadata = []
    for task in tasks:
        level, desc = extract_complexity(task)
        domain = extract_domain(task)
        schema = task.get("target_schema", {})
        title = schema.get("title", "")
        field_types = extract_field_types(task)
        metadata.append({
            "task_id": task.get("id", ""),
            "complexity": level,
            "complexity_desc": desc,
            "domain": domain,
            "title": title,
            "field_types": field_types,
        })

    output = {
        "model": MODEL_NAME,
        "count": len(vectors),
        "dim": len(vectors[0]),
        "vectors": vectors,
        "metadata": metadata,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
