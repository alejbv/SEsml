"""
Verification layer for the Extraction v2 pipeline.

POST Validation (deterministic):
  Layer 1 — Pydantic model_validate_json (structural).
  Layer 2 — Grounding string matching (values in source text).
  Layer 3 — Business rules (@model_validator equivalent).
"""

import json
import re
from logging import getLogger
from typing import Any

logger = getLogger("gensie")


# ── POST Validation ──────────────────────────────────────────────


def validate_post(
    result_json: dict[str, Any],
    schema: dict[str, Any],
    input_text: str,
    *,
    pydantic_model: type | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Run deterministic POST validation layers.

    Layer 1: Structural validation (types, enums, required).
    Layer 2: Grounding (string values exist in source text).
    Layer 3: Business rules (if a Pydantic model with validators is provided).

    Args:
        result_json: The JSON dict to validate.
        schema: Target JSON Schema (optimised).
        input_text: Original source text.
        pydantic_model: Optional Pydantic model for ``model_validate``
            (Layer 3). If provided, business rules in ``@model_validator``
            are checked.

    Returns:
        Tuple of (is_valid, error_messages, possibly_corrected_json).
    """
    errors: list[str] = []

    # ── Layer 1: Structural ─────────────────────────────────────────
    layer1_errors = _check_structure(result_json, schema)
    errors.extend(layer1_errors)

    # ── Layer 2: Grounding ──────────────────────────────────────────
    layer2_errors = _check_grounding(result_json, input_text, schema)
    errors.extend(layer2_errors)

    # ── Layer 3: Pydantic business rules ────────────────────────────
    corrected = dict(result_json)
    if pydantic_model is not None and not errors:
        try:
            validated = pydantic_model.model_validate(result_json)
            corrected = validated.model_dump(mode="json")
        except Exception as e:
            errors.append(f"Pydantic validation error: {e}")

    return len(errors) == 0, errors, corrected


def _check_structure(
    result: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """Layer 1: Validate types, enums, and required fields."""
    errors: list[str] = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        if field not in result or result[field] is None:
            errors.append(f"Campo requerido '{field}' es null o ausente")

    for field, value in result.items():
        if value is None:
            continue

        props = properties.get(field, {})
        if not props:
            continue

        # Enum check
        if "enum" in props and value not in props["enum"]:
            errors.append(
                f"Campo '{field}': valor '{value}' no está en enum {props['enum']}"
            )

        # Type check
        field_type = _resolve_prop_type(props)
        if field_type == "integer" and not isinstance(value, int):
            if isinstance(value, bool):
                errors.append(f"Campo '{field}': bool no es integer válido")
            elif not isinstance(value, (int, float)):
                errors.append(
                    f"Campo '{field}': esperaba integer, obtuvo {type(value).__name__}"
                )
        elif field_type == "number" and not isinstance(value, (int, float)):
            errors.append(
                f"Campo '{field}': esperaba number, obtuvo {type(value).__name__}"
            )
        elif field_type == "boolean" and not isinstance(value, bool):
            errors.append(
                f"Campo '{field}': esperaba boolean, obtuvo {type(value).__name__}"
            )
        elif field_type == "array" and not isinstance(value, list):
            errors.append(
                f"Campo '{field}': esperaba array, obtuvo {type(value).__name__}"
            )

    return errors


def _check_grounding(
    result: dict[str, Any],
    input_text: str,
    schema: dict[str, Any],
) -> list[str]:
    """Layer 2: Verify string values exist in source text.

    Skips:
    * Fields with ``enum`` (categorical, not literal).
    * Values matching date patterns (``YYYY-MM-DD``).
    * Values matching numeric patterns.
    """
    errors: list[str] = []
    input_normalized = _normalize_text(input_text)
    # Date pattern: YYYY-MM-DD or similar
    _DATE_PATTERN = re.compile(r"^\d{2,4}[-\/]\d{1,2}[-\/]\d{1,4}$")
    # Numeric pattern
    _NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")

    for field, value in result.items():
        if value is None:
            continue
        if not isinstance(value, str):
            continue

        # Skip enum fields (categorías, no literales)
        props = schema.get("properties", {}).get(field, {})
        if "enum" in props:
            continue

        # Skip normalized dates and numbers
        if _DATE_PATTERN.match(value) or _NUMERIC_PATTERN.match(value):
            continue

        value_normalized = _normalize_text(value)
        if value_normalized in input_normalized:
            continue

        # Token overlap fallback
        value_tokens = set(value_normalized.split())
        input_tokens = set(input_normalized.split())
        overlap = len(value_tokens & input_tokens) / max(len(value_tokens), 1)
        if overlap < 0.5:
            errors.append(
                f"Campo '{field}': valor '{value}' no encontrado en el texto (grounding)"
            )

    return errors


def _resolve_prop_type(props: dict[str, Any]) -> str:
    """Resolve the effective type of a schema property."""
    if "enum" in props:
        return "string"

    raw = props.get("type")
    if isinstance(raw, str):
        return raw

    # Handle anyOf with null
    for alt in props.get("anyOf", []):
        if isinstance(alt, dict) and alt.get("type") != "null":
            return alt.get("type", "string")

    # Handle type list
    if isinstance(raw, list):
        non_null = [t for t in raw if t != "null"]
        return non_null[0] if non_null else "string"

    return "string"


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, no accents, single spaces."""
    text = text.lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ü": "u", "ñ": "n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
