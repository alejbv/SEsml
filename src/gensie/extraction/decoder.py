"""
Phase: Constrained Decoding (Draft-as-Context).

Takes a semi-structured draft (``campo: valor`` lines) + optimised schema
and produces the final JSON via ``response_format=json_schema``.

The guiding principle is:
  - Schema defines the STRUCTURE (types, required, enums).
  - Draft defines the VALUES (extracted content).
  - The model maps draft values into schema structure without collisions.

Key differences from the qaloop decoder:
  * Uses SchemaOptimizer's output (no ``$ref``, no ``default``).
  * Does NOT set ``strict=True`` (LM Studio / llama.cpp may not support it).
  * Includes the draft as context in the prompt (Draft-as-Context).
  * Falls back to local semi-structured parsing if structured output fails.
"""

import json
import re
from logging import getLogger
from typing import Any

from .prompts import CD_DRAFT_CONTEXT

logger = getLogger("gensie")


def _format_draft(draft: str | dict[str, Any] | None) -> str:
    """Normalise a draft value to a string for the prompt.

    * If it is a ``dict`` → JSON-dump with indentation.
    * If it is a ``str`` → return as-is (the draft may be semi-structured).
    * If ``None`` → return "(sin borrador)" so the decoder knows there's no context.
    """
    if draft is None:
        return "(sin borrador)"
    if isinstance(draft, dict):
        return json.dumps(draft, indent=2, ensure_ascii=False)
    return draft


def _parse_kv_draft(text: str) -> dict[str, Any]:
    """Parse semi-structured ``campo: valor`` lines into a plain dict.

    Handles:
      - ``campo: valor``  → basic key-value
      - ``null``, ``true``, ``false`` → Python None, True, False
      - numbers → int/float
      - comma-separated values → kept as string (the decoder handles formatting)
    """
    result: dict[str, Any] = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, raw_value = line.partition(": ")
        key = key.strip()
        value: Any = raw_value.strip()

        # Coerce literals
        if value == "null":
            value = None
        elif value == "true":
            value = True
        elif value == "false":
            value = False
        else:
            # Try number
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except (ValueError, TypeError):
                pass  # keep as string

        result[key] = value
    return result


def _extract_json(content: str) -> dict[str, Any]:
    """Extract a JSON dict from a raw string, trying multiple strategies.

    Tries (in order):
      1. Direct ``json.loads`` on the whole string.
      2. Extract from a ```json```` markdown block.
      3. Extract the first top-level ``{…}`` object.
      4. Parse as semi-structured ``campo: valor`` lines (new).
    """
    text = content.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown block
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. First { … }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    # 4. Semi-structured KV lines (new)
    kv = _parse_kv_draft(text)
    if kv:
        return kv

    raise ValueError(f"No se pudo extraer JSON de:\n{content[:300]}...")


def decode_to_json(
    client: Any,
    model: str,
    draft: str | dict[str, Any] | None,
    schema: dict[str, Any],
    *,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Convert draft into final validated JSON via constrained decoding.

    Makes exactly **one** LLM call with ``response_format=json_schema``.
    If that fails, falls back to parsing the draft locally (no LLM).

    The ``draft`` argument can be a ``dict``, a **raw string** (semi-structured),
    or ``None`` (no draft available).

    Args:
        client: OpenAI-compatible client.
        model: Model name.
        draft: Draft output from DraftEngine (dict, raw string, or None).
        schema: Optimised JSON Schema (from SchemaOptimizer.optimized_schema).
        max_tokens: Max completion tokens.

    Returns:
        Final extracted JSON as dict.

    Raises:
        ValueError: If JSON cannot be extracted from the draft.
    """
    prompt = CD_DRAFT_CONTEXT.format(
        draft_json=_format_draft(draft),
    )

    # ── Attempt 1: Structured output (constrained decoding) ────────────
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": schema,
                },
            },
        )
        result = _extract_json(response.choices[0].message.content or "")
        logger.info("[decoder] Structured output succeeded.")
        return result

    except Exception as e:
        logger.warning(
            f"[decoder] Structured output failed: {e}. "
            "Falling back to local extraction."
        )

    # ── Local fallback: extract from the draft (no LLM) ────────────────
    if isinstance(draft, dict):
        return draft
    if draft is None:
        raise ValueError("No draft available and structured output failed.")
    return _extract_json(draft)
