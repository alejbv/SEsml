"""
SchemaOptimizer — Pydantic BaseModel → Optimized JSON Schema + Field Metadata.

PRE-optimization layer (E1 in the definitive strategy).
Takes a Pydantic BaseModel as single source of truth and produces:

  FieldMetadata: structured list of all fields with type, description,
      enum_values, optionality, defaults, nesting info.
  Optimized JSON Schema: flattened, $ref-resolved, strict-mode-compatible
      schema ready for constrained decoding engines (OpenAI, Outlines, etc.).

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel


# ──────────────────────────────────────────────────────────────────────────
# 1.a — Field Metadata
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class FieldMetadata:
    """Structured metadata for a single field in the extraction schema.

    Attributes:
        name: Short field name (e.g. ``"contract_title"``).
        path: Dot-separated path from root (e.g. ``"parties"``,
            ``"symptoms.name"``).
        description: Human-readable description from ``Field(description=...)``.
        field_type: Normalised type — one of ``"string"``, ``"integer"``,
            ``"number"``, ``"boolean"``, ``"enum"``, ``"array"``, ``"object"``.
        is_optional: ``True`` if the schema allows ``null`` (``anyOf`` with
            ``{"type": "null"}``).
        default: The default value from the schema (``None`` if not set).
        enum_values: List of allowed values for ``enum``-typed fields,
            ``None`` otherwise.
        item_type: Item type for ``array``-typed fields (e.g. ``"string"``),
            ``None`` otherwise.
        nested_fields: Child fields for ``object``-typed fields, ``None``
            otherwise.
    """

    name: str
    path: str
    description: str
    field_type: str
    is_optional: bool = False
    default: Any = None
    enum_values: Optional[list[str]] = None
    item_type: Optional[str] = None
    nested_fields: Optional[list[FieldMetadata]] = None


# ──────────────────────────────────────────────────────────────────────────
# 1.b — Schema Optimizer
# ──────────────────────────────────────────────────────────────────────────


class SchemaOptimizer:
    """Transforms a Pydantic ``BaseModel`` into assets ready for the pipeline.

    Typical usage::

        optimizer = SchemaOptimizer(MyExtractionModel)
        schema_opt = optimizer.optimized_schema       # 1.b → CD engine
        metadata   = optimizer.field_metadata          # 1.a → draft prompts
    """

    # Keys that constrained-decoding engines **reject**.
    # We strip these recursively from every schema node.
    _UNSAFE_KEYS = frozenset({
        "default",
        "title",
        "format",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "$defs",
    })

    def __init__(self, model: type[BaseModel]) -> None:
        self._model = model
        # Raw schema from Pydantic — the single source of truth.
        raw: dict[str, Any] = model.model_json_schema(
            union_format="smart",  # consistent anyOf handling
        )
        # Resolve all $ref references *once*.
        defs: dict[str, Any] = raw.pop("$defs", {})
        self._resolved: dict[str, Any] = self._resolve_refs(raw, defs)

        # Build the two assets (lazy).
        self._schema: dict[str, Any] | None = None
        self._metadata: list[FieldMetadata] | None = None

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def optimized_schema(self) -> dict[str, Any]:
        """1.b: Optimised JSON Schema ready for constrained decoding.

        Guarantees:
        * All ``$ref`` / ``$defs`` resolved inline.
        * ``additionalProperties: false`` on **every** object.
        * **All** properties listed in ``required`` (needed for CD).
        * ``default``, ``title``, ``format``, numeric constraints,
          and ``$defs`` stripped.
        * ``enum`` and ``description`` preserved (critical for quality).
        * ``anyOf``/``oneOf`` with ``null`` preserved — the model may output
          ``null`` for optional fields.
        """
        if self._schema is not None:
            return self._schema

        result = self._prepare_strict(self._resolved)
        self._enforce_additional_properties(result)
        self._force_required(result)
        result.pop("title", None)
        result.pop("$defs", None)

        self._schema = result
        return self._schema

    @property
    def field_metadata(self) -> list[FieldMetadata]:
        """1.a: Structured metadata for every field (recursive)."""
        if self._metadata is not None:
            return self._metadata
        self._metadata = self._extract_metadata(self._resolved)
        return self._metadata

    @classmethod
    def from_raw_schema(cls, raw_schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve and clean a raw JSON Schema dict (no Pydantic model).

        Useful when the pipeline receives a dict schema instead of a model.
        Returns an optimized schema dict directly.
        """
        defs = raw_schema.pop("$defs", {})
        resolved = cls._resolve_refs(raw_schema, defs)
        cleaned = cls._prepare_strict(resolved)
        cls._enforce_additional_properties(cleaned)
        cls._force_required(cleaned)
        cleaned.pop("title", None)
        cleaned.pop("$defs", None)
        return cleaned

    # ── $ref resolution (single implementation) ─────────────────────────

    @staticmethod
    def _resolve_refs(obj: Any, defs: dict[str, Any]) -> Any:
        """Recursively resolve ``$ref`` references **inline**.

        Handles both ``#/$defs/Name`` and ``#/...`` paths.
        When a ``$ref`` has sibling keys (e.g. ``description``), those
        siblings are **merged** into the resolved definition.
        """
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path: str = obj["$ref"]
                resolved: dict[str, Any] = {}

                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    resolved = defs.get(def_name, {})
                elif ref_path.startswith("#/"):
                    parts = ref_path[2:].split("/")
                    resolved = defs
                    for p in parts:
                        if isinstance(resolved, dict):
                            resolved = resolved.get(p, {})
                        else:
                            resolved = {}
                            break
                # Merge sibling keys on top of the resolved definition.
                merged: dict[str, Any] = dict(resolved)
                for k, v in obj.items():
                    if k != "$ref":
                        merged[k] = v
                return SchemaOptimizer._resolve_refs(merged, defs)

            return {
                k: SchemaOptimizer._resolve_refs(v, defs)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [SchemaOptimizer._resolve_refs(item, defs) for item in obj]

        return obj

    # ── Schema cleaning ─────────────────────────────────────────────────

    @staticmethod
    def _prepare_strict(schema: dict[str, Any]) -> dict[str, Any]:
        """Remove keys that constrained-decoding engines reject.

        Strips ``default``, ``title``, ``format``, numeric constraints, etc.
        from the entire schema tree while preserving all structural keys
        and dynamic property names.

        Note: Keys inside ``properties`` are **user-defined field names**,
        not schema keywords — they are never stripped.
        """
        result: dict[str, Any] = {}
        for k, v in schema.items():
            if k in SchemaOptimizer._UNSAFE_KEYS:
                continue
            if k == "properties" and isinstance(v, dict):
                # Property names are user-defined field names (e.g., "title",
                # "default", "format") — never strip them. Recurse into values.
                result[k] = {
                    prop_name: (
                        SchemaOptimizer._prepare_strict(prop_schema)
                        if isinstance(prop_schema, dict)
                        else prop_schema
                    )
                    for prop_name, prop_schema in v.items()
                }
            elif isinstance(v, dict):
                result[k] = SchemaOptimizer._prepare_strict(v)
            elif isinstance(v, list):
                result[k] = [
                    SchemaOptimizer._prepare_strict(i) if isinstance(i, dict) else i
                    for i in v
                ]
            else:
                result[k] = v
        return result

    @staticmethod
    def _enforce_additional_properties(schema: dict[str, Any]) -> None:
        """Set ``additionalProperties: false`` on **every** object."""
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            for prop in schema.get("properties", {}).values():
                if isinstance(prop, dict):
                    SchemaOptimizer._enforce_additional_properties(prop)
        items = schema.get("items")
        if isinstance(items, dict):
            SchemaOptimizer._enforce_additional_properties(items)

    @staticmethod
    def _force_required(schema: dict[str, Any]) -> None:
        """Force all properties into ``required``.

        Optional fields keep their ``anyOf`` with ``null`` so the model can
        still output ``null``.
        """
        if schema.get("type") != "object":
            return

        props: dict[str, Any] = schema.get("properties", {})
        required: list[str] = []

        for prop_name, prop_schema in props.items():
            if not isinstance(prop_schema, dict):
                continue
            required.append(prop_name)
            SchemaOptimizer._force_required(prop_schema)
            items = prop_schema.get("items")
            if isinstance(items, dict):
                SchemaOptimizer._force_required(items)

        schema["required"] = required

    # ── Field metadata extraction ───────────────────────────────────────

    @staticmethod
    def _extract_metadata(
        schema: dict[str, Any],
        prefix: str = "",
    ) -> list[FieldMetadata]:
        """Recursively extract ``FieldMetadata`` from a resolved schema."""
        if schema.get("type") != "object":
            return []

        props: dict[str, Any] = schema.get("properties", {})
        required_set = set(schema.get("required", []))

        fields: list[FieldMetadata] = []
        for name, prop in props.items():
            if not isinstance(prop, dict):
                continue

            path = f"{prefix}.{name}" if prefix else name
            metadata = SchemaOptimizer._describe_field(name, path, prop, required_set)
            fields.append(metadata)

            if metadata.field_type == "object":
                metadata.nested_fields = SchemaOptimizer._extract_metadata(
                    prop, prefix=path,
                )
            if metadata.field_type == "array" and isinstance(prop.get("items"), dict):
                item_schema: dict[str, Any] = prop["items"]
                if item_schema.get("type") == "object":
                    nested = SchemaOptimizer._extract_metadata(
                        item_schema, prefix=f"{path}[]",
                    )
                    if metadata.nested_fields is None:
                        metadata.nested_fields = []
                    metadata.nested_fields.extend(nested)

        return fields

    @classmethod
    def _describe_field(
        cls,
        name: str,
        path: str,
        prop: dict[str, Any],
        required_set: set[str],
    ) -> FieldMetadata:
        """Build a ``FieldMetadata`` for a single property."""
        field_type = cls._resolve_type(prop)
        enum_values: list[str] | None = prop.get("enum")
        description: str = prop.get("description", name)
        is_optional = name not in required_set
        default = prop.get("default")
        item_type: str | None = None

        if field_type == "enum" and enum_values is None:
            field_type = "string"

        if field_type == "array":
            items = prop.get("items", {})
            if isinstance(items, dict):
                item_type = cls._resolve_type(items)

        return FieldMetadata(
            name=name,
            path=path,
            description=description,
            field_type=field_type,
            is_optional=is_optional,
            default=default,
            enum_values=enum_values,
            item_type=item_type,
        )

    @staticmethod
    def _resolve_type(prop: dict[str, Any]) -> str:
        """Resolve the logical type of a schema property."""
        if "enum" in prop:
            return "enum"

        raw_type = prop.get("type")
        if isinstance(raw_type, str):
            return raw_type

        if isinstance(raw_type, list):
            non_null = [t for t in raw_type if t != "null"]
            return non_null[0] if non_null else "string"

        for key in ("anyOf", "oneOf"):
            alternatives = prop.get(key)
            if isinstance(alternatives, list):
                for alt in alternatives:
                    if isinstance(alt, dict) and alt.get("type") != "null":
                        return SchemaOptimizer._resolve_type(alt)

        return "string"  # best-effort fallback

    # ── Utility ─────────────────────────────────────────────────────────

    def pretty_print_schema(self) -> str:
        """Return the optimised schema as a pretty-printed JSON string."""
        return json.dumps(self.optimized_schema, indent=2)

    def pretty_print_metadata(self) -> str:
        """Return the field metadata as a human-readable table."""
        lines: list[str] = []
        for md in self._flatten_metadata(self.field_metadata):
            opt = "optional" if md.is_optional else "required"
            enum_info = f" enum={md.enum_values}" if md.enum_values else ""
            item_info = f" item_type={md.item_type}" if md.item_type else ""
            lines.append(
                f"  {md.path:<40} {md.field_type:<10} {opt:<10}"
                f"{enum_info}{item_info}"
            )
        return "\n".join(lines) if lines else "  (no fields)"

    @staticmethod
    def _flatten_metadata(fields: list[FieldMetadata]) -> list[FieldMetadata]:
        """Flatten nested metadata into a single list."""
        result: list[FieldMetadata] = []
        for md in fields:
            result.append(md)
            if md.nested_fields:
                result.extend(SchemaOptimizer._flatten_metadata(md.nested_fields))
        return result
