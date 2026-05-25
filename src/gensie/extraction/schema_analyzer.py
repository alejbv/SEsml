"""
SchemaAnalyzer — Profile a task for adaptive prompt assembly.

Takes a task + precomputed dev vectors and produces a ``TaskProfile``:

  1. Build fingerprint for the current task.
  2. Embed it with the same model used for dev vectors.
  3. Find top-K similar dev tasks via cosine similarity.
  4. Determine complexity via weighted voting.
  5. Extract field types from the current schema.
  6. Determine domain from similar tasks.
  7. Detect known error patterns.

This module does NOT call any LLM — purely local computation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from logging import getLogger

logger = getLogger("gensie")

# ── Default path for precomputed vectors ──────────────
_HERE = Path(__file__).resolve().parent
_DEFAULT_VECTORS_PATH = _HERE / "data" / "dev_vectors.json"
_DEFAULT_ERROR_MAP_PATH = _HERE / "data" / "error_map.json"

# ── Cosine similarity threshold for pattern errors ─────
_PATTERN_THRESHOLD = 0.5
_DOMAIN_THRESHOLD = 0.4
_FEW_SHOT_THRESHOLD = 0.75

# ── Number of neighbors for voting ────────────────────
K_NEIGHBORS = 15


# ═══════════════════════════════════════════════════════
#  TaskProfile
# ═══════════════════════════════════════════════════════


@dataclass
class TaskProfile:
    """Profile summarising a task for the PromptAssembler.

    Attributes:
        complexity: Complexity level (L1–L9) from weighted voting.
        complexity_desc: Human-readable description (e.g. "Hierarchical Grouping").
        complexity_confidence: P_selected — how confident the vote is (0–1).
        field_types: Sorted list of types present in the schema
                      (e.g. ["array", "enum", "object"]).
        has_nullable: Whether any field is nullable.
        domain: Detected domain from similar tasks.
        domain_confidence: Proportion of neighbors with same domain.
        pattern: Schema title (e.g. "DiseaseProfile").
        has_known_errors: Whether this pattern has known error rules.
        error_rules: List of error rule strings for this pattern.
        closest_task_id: ID of the most similar dev task.
        closest_similarity: Cosine similarity of the closest task.
        avg_similarity: Average similarity across top-K.
    """
    complexity: str = "L4"
    complexity_desc: str = "General"
    complexity_confidence: float = 0.0
    field_types: list[str] = field(default_factory=list)
    has_nullable: bool = False
    domain: str = ""
    domain_confidence: float = 0.0
    pattern: str = ""
    has_known_errors: bool = False
    error_rules: list[str] = field(default_factory=list)
    closest_task_id: str = ""
    closest_similarity: float = 0.0
    avg_similarity: float = 0.0
    _neighbors: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
#  SchemaAnalyzer
# ═══════════════════════════════════════════════════════


class SchemaAnalyzer:
    """Analyse a task and produce a TaskProfile.

    Usage::

        analyzer = SchemaAnalyzer()
        profile = analyzer.analyze(task_dict)
    """

    def __init__(
        self,
        vectors_path: str | Path = _DEFAULT_VECTORS_PATH,
        error_map_path: str | Path = _DEFAULT_ERROR_MAP_PATH,
        k: int = K_NEIGHBORS,
    ):
        self.k = k
        self._vectors_data = self._load_vectors(vectors_path)
        self._error_map = self._load_error_map(error_map_path)
        self._model = None  # lazy import/init

        # Extract arrays for fast computation
        self._dev_vectors = self._vectors_data["vectors"]
        self._dev_metadata = self._vectors_data["metadata"]

    # ── Public API ───────────────────────────────────────

    def analyze(self, task: dict[str, Any]) -> TaskProfile:
        """Produce a TaskProfile for the given task."""
        fingerprint = self._build_fingerprint(task)
        query_vector = self._embed([fingerprint])[0]

        # ── 1. Semantic search ────────────────────────────
        similarities = self._cosine_similarities(query_vector, self._dev_vectors)
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )[:self.k]

        neighbors = []
        for idx in top_indices:
            neighbors.append({
                "task_id": self._dev_metadata[idx]["task_id"],
                "similarity": float(similarities[idx]),
                "complexity": self._dev_metadata[idx]["complexity"],
                "complexity_desc": self._dev_metadata[idx]["complexity_desc"],
                "domain": self._dev_metadata[idx]["domain"],
                "title": self._dev_metadata[idx]["title"],
                "field_types": self._dev_metadata[idx]["field_types"],
            })

        avg_sim = sum(n["similarity"] for n in neighbors) / len(neighbors)
        closest = neighbors[0] if neighbors else {}

        # ── 2. Complexity via weighted voting ─────────────
        complexity, complexity_conf, complexity_desc = self._vote_complexity(neighbors)

        # ── 3. Field types from current schema ────────────
        field_types = self._detect_field_types(task)
        has_nullable = self._has_nullable(task)

        # ── 4. Domain from neighbors ──────────────────────
        domain, domain_conf = self._detect_domain(neighbors)

        # ── 5. Pattern & error rules ──────────────────────
        pattern = task.get("target_schema", {}).get("title", "")
        error_rules = self._error_map.get(pattern, {}).get("rules", [])
        has_known_errors = (
            len(error_rules) > 0
            and closest.get("similarity", 0) >= _PATTERN_THRESHOLD
        )

        return TaskProfile(
            complexity=complexity,
            complexity_desc=complexity_desc,
            complexity_confidence=complexity_conf,
            field_types=field_types,
            has_nullable=has_nullable,
            domain=domain,
            domain_confidence=domain_conf,
            pattern=pattern,
            has_known_errors=has_known_errors,
            error_rules=error_rules[:5] if has_known_errors else [],
            closest_task_id=closest.get("task_id", ""),
            closest_similarity=closest.get("similarity", 0.0),
            avg_similarity=avg_sim,
            _neighbors=neighbors,
        )

    # ── Fingerprint & Embedding ──────────────────────────

    @staticmethod
    def _build_fingerprint(task: dict[str, Any]) -> str:
        """Build text fingerprint for a task."""
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

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the cached model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed not installed. Run: uv sync --group dev"
                ) from exc
            model_name = self._vectors_data.get("model", "BAAI/bge-small-en-v1.5")
            self._model = TextEmbedding(model_name)
        return [emb.tolist() for emb in self._model.embed(texts)]

    # ── Similarity ────────────────────────────────────────

    @staticmethod
    def _cosine_similarities(
        query: list[float],
        vectors: list[list[float]],
    ) -> list[float]:
        """Compute cosine similarity between query and each vector."""
        q_norm = math.sqrt(sum(v * v for v in query))
        if q_norm == 0:
            return [0.0] * len(vectors)

        results = []
        for vec in vectors:
            dot = sum(q * v for q, v in zip(query, vec))
            v_norm = math.sqrt(sum(v * v for v in vec))
            if v_norm == 0:
                results.append(0.0)
            else:
                results.append(dot / (q_norm * v_norm))
        return results

    # ── Voting ────────────────────────────────────────────

    @staticmethod
    def _vote_complexity(
        neighbors: list[dict],
    ) -> tuple[str, float, str]:
        """Weighted voting for complexity level.

        Returns (winning_level, confidence, description).
        """
        scores: dict[str, float] = {}
        descs: dict[str, str] = {}

        for n in neighbors:
            w = n["similarity"] ** 2  # quadratic weighting
            level = n["complexity"]
            scores[level] = scores.get(level, 0.0) + w
            descs[level] = n.get("complexity_desc", "")

        total = sum(scores.values())
        if total == 0:
            return "L4", 0.0, "General"

        winner = max(scores, key=scores.get)
        confidence = scores[winner] / total
        desc = descs.get(winner, "General")
        return winner, round(confidence, 4), desc

    @staticmethod
    def _detect_domain(neighbors: list[dict]) -> tuple[str, float]:
        """Detect domain from neighbor majority."""
        domain_counts: dict[str, float] = {}
        for n in neighbors:
            d = n.get("domain", "")
            if d:
                domain_counts[d] = domain_counts.get(d, 0) + n["similarity"]

        if not domain_counts:
            return "", 0.0

        winner = max(domain_counts, key=domain_counts.get)
        total = sum(domain_counts.values())
        confidence = domain_counts[winner] / total if total > 0 else 0.0
        return winner, round(confidence, 4)

    # ── Schema analysis ──────────────────────────────────

    @staticmethod
    def _detect_field_types(task: dict[str, Any]) -> list[str]:
        """Detect types of fields present in the schema."""
        props = task.get("target_schema", {}).get("properties", {})
        types = set()
        for name, prop in props.items():
            if not isinstance(prop, dict):
                continue
            ft = _resolve_type_simple(prop)
            types.add(ft)
            if ft == "array":
                items = prop.get("items", {})
                if isinstance(items, dict) and items.get("type") == "object":
                    types.add("nested_object")
            elif ft == "object":
                types.add("nested_object")
            elif ft == "string" and "enum" in prop:
                types.add("enum")
        # Normalise ordering
        priority = ["array", "enum", "nested_object", "number", "integer", "boolean", "string"]
        ordered = [t for t in priority if t in types]
        for t in sorted(types):
            if t not in ordered:
                ordered.append(t)
        return ordered

    @staticmethod
    def _has_nullable(task: dict[str, Any]) -> bool:
        """Check if any property in the schema is nullable."""
        props = task.get("target_schema", {}).get("properties", {})
        for name, prop in props.items():
            if not isinstance(prop, dict):
                continue
            raw_type = prop.get("type")
            if isinstance(raw_type, list) and "null" in raw_type:
                return True
            for key in ("anyOf", "oneOf"):
                alts = prop.get(key)
                if isinstance(alts, list):
                    for alt in alts:
                        if isinstance(alt, dict) and alt.get("type") == "null":
                            return True
        return False

    # ── Loaders ──────────────────────────────────────────

    @staticmethod
    def _load_vectors(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Precomputed vectors not found at {path}. "
                "Run: python scripts/precompute_vectors.py"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_error_map(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            logger.warning("Error map not found at %s. Using empty map.", path)
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)


# ── Helpers ─────────────────────────────────────────────


def _resolve_type_simple(prop: dict) -> str:
    """Resolve the logical type of a schema property (simplified)."""
    if "enum" in prop:
        return "enum"
    raw_type = prop.get("type")
    if isinstance(raw_type, str):
        return raw_type
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        return non_null[0] if non_null else "string"
    for key in ("anyOf", "oneOf"):
        alts = prop.get(key)
        if isinstance(alts, list):
            for alt in alts:
                if isinstance(alt, dict) and alt.get("type") != "null":
                    return _resolve_type_simple(alt)
    return "string"
