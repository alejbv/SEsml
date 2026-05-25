"""
PromptAssembler — Build adaptive prompts from a TaskProfile.

Takes a ``TaskProfile`` and assembles a modular prompt within a char budget.

Module order (highest priority → lowest):
  1. BASE       — always included (~280 chars)
  2. COMPLEXITY — included if P_selected > 0.4 (~160 chars)
  3. FIELD_TYPES — one per type present in schema (~90–120 chars each)
  4. PATTERN_ERRORS — included if pattern has known errors and sim ≥ 0.5 (~180 chars)
  5. DOMAIN_TIPS — included if domain_confidence ≥ 0.4 (~120 chars)
  6. FEW_SHOT — included only if closest_similarity ≥ 0.75 (~250 chars)

Total budget target: ~1400 chars for the instruction part (template excluded).
"""

from __future__ import annotations

from typing import Any

from .schema_analyzer import TaskProfile

# ── Budget ──────────────────────────────────────────────
_BUDGET = 1400  # target chars for instruction modules

# ── Module thresholds ───────────────────────────────────
_THRESHOLD_COMPLEXITY = 0.4
_THRESHOLD_PATTERN = 0.5
_THRESHOLD_DOMAIN = 0.4
_THRESHOLD_FEW_SHOT = 0.75

# ═══════════════════════════════════════════════════════
#  Module templates
# ═══════════════════════════════════════════════════════

BASE = (
    "\n"
    "Para CADA campo, busca evidencia en el texto y clasifícala:\n"
    "\n"
    "LITERAL → el valor aparece textualmente → cópialo exacto.\n"
    "DESCRIPTIVA → el texto describe el concepto → mapea al esquema.\n"
    "INFERIDA → combina ≥2 pistas textuales → razona en <Thinking>.\n"
    "AGREGADA (arrays) → reúne info dispersa en el texto.\n"
    "\n"
    "[INFERENCIA]\n"
    "  Prohibido inferir: idioma, nacionalidad, fecha de publicación,\n"
    "  ni ningún metadato externo. La inferencia solo combina\n"
    "  EVIDENCIA TEXTUAL DIRECTA.\n"
    "\n"
    "[FORMATO]\n"
    "  Para CADA campo, lee su description en el esquema.\n"
    "  la description especifica el formato esperado.\n"
    "  SIGUE la description. NO uses reglas de formato fijas."
)

COMPLEXITY_MODULES = {
    "L1": (
        "\n\n[COMPLEJIDAD L1 — Simple]\n"
        "Campos planos e independientes. Cada valor se extrae directamente del texto."
    ),
    "L2": (
        "\n\n[COMPLEJIDAD L2 — Múltiples valores]\n"
        "Puede haber múltiples instancias del mismo tipo. Enumera cada una por separado."
    ),
    "L3": (
        "\n\n[COMPLEJIDAD L3 — Anidado]\n"
        "Respeta objetos dentro de objetos. No aplastes jerarquías."
    ),
    "L4": (
        "\n\n[COMPLEJIDAD L4 — Campos cruzados]\n"
        "Algunos campos dependen de otros. Verifica consistencia entre campos relacionados."
    ),
    "L5": (
        "\n\n[COMPLEJIDAD L5 — Jerárquico]\n"
        "Objetos anidados con arrays de objetos. Sigue exactamente la anidación del esquema."
    ),
    "L7": (
        "\n\n[COMPLEJIDAD L7 — Agregación dispersa]\n"
        "Información dispersa en múltiples secciones. Recopila datos de TODO el texto."
    ),
    "L8": (
        "\n\n[COMPLEJIDAD L8 — Listas anidadas + Mapeo]\n"
        "Arrays de objetos con subcampos y mapeo semántico a enums."
    ),
    "L9": (
        "\n\n[COMPLEJIDAD L9 — Null traps]\n"
        "Campos con valor null frecuente. Solo extrae valores EXPLÍCITAMENTE mencionados."
    ),
    "L10": (
        "\n\n[COMPLEJIDAD L10 — Razonamiento holístico]\n"
        "Requiere entender el texto como un todo. Sintetiza información\n"
        "de múltiples secciones para construir cada campo del esquema."
    ),
}

# Default when confidence is low
_COMPLEXITY_DEFAULT = (
    "\n\n[COMPLEJIDAD MIXTA]\n"
    "Schema de complejidad variada. Sigue la estructura exacta del esquema."
)

FIELD_RULES = {
    "array": (
        "\n\n[ARRAYS]\n"
        "Identifica TODOS los elementos. Cuéntalos mentalmente.\n"
        "Cada entidad en su propio elemento. 0 elementos → []."
    ),
    "enum": (
        "\n\n[ENUMS]\n"
        "Mapea el texto al valor EXACTO del enum (case-sensitive).\n"
        "NO uses sinónimos ni copies texto literal en un enum."
    ),
    "nested_object": (
        "\n\n[OBJETOS ANIDADOS]\n"
        "Respeta la estructura exacta de objetos dentro de objetos.\n"
        "Cada nivel de anidación es independiente."
    ),
    "boolean": (
        "\n\n[BOOLEANOS]\n"
        "Solo true si el texto afirma explícitamente el hecho.\n"
        "Sin afirmación explícita → null. NO infieras."
    ),
    "number": (
        "\n\n[NUMÉRICOS]\n"
        "El número debe estar EXPLÍCITAMENTE en el texto.\n"
        "Si no hay número explícito → null."
    ),
    "integer": (
        "\n\n[NUMÉRICOS]\n"
        "El número debe estar EXPLÍCITAMENTE en el texto.\n"
        "Si no hay número explícito → null."
    ),
    "_seen_numeric": False,  # marker to deduplicate; not a rule
}

NULLABLE_RULE = (
    "\n\n[CAMPOS OPCIONALES]\n"
    "Campos que pueden ser null si no hay evidencia.\n"
    "null = AUSENCIA. No \"\" ni \"null\". No inventes valores por defecto."
)

DOMAIN_TIPS = {
    "legal": (
        "\n\n[DOMINIO LEGAL]\n"
        "Busca cláusulas numeradas, fechas explícitas, montos en números.\n"
        "Las partes del contrato son organizaciones, no personas físicas."
    ),
    "medical": (
        "\n\n[DOMINIO MÉDICO]\n"
        "Síntomas como array de objetos. Enfermedad como string literal.\n"
        "Severidad solo si está explícita. Diagnóstico como string descriptivo."
    ),
    "stem": (
        "\n\n[DOMINIO CIENTÍFICO]\n"
        "Valores numéricos con unidades. Fechas de descubrimiento explícitas.\n"
        "Nombres propios de cuerpos celestes o conceptos."
    ),
    "cultural": (
        "\n\n[DOMINIO CULTURAL]\n"
        "Nombres de obras, autores, fechas de publicación.\n"
        "Clasificaciones por género o tipo."
    ),
    "technical": (
        "\n\n[DOMINIO TÉCNICO]\n"
        "Software, versiones, sistemas operativos, lenguajes.\n"
        "Extrae valores exactos sin modificaciones."
    ),
    "general": (
        "\n\n[DOMINIO GENERAL]\n"
        "Eventos, desastres, noticias. Fechas, ubicaciones, afectados.\n"
        "Cifras numéricas explícitas."
    ),
    "lifestyle": (
        "\n\n[DOMINIO COTIDIANO]\n"
        "Recetas, ingredientes, pasos. Respeta el orden de las instrucciones."
    ),
}

DOMAIN_DEFAULT = ""

# ── Output format ────────────────────────────────────────
_OUTPUT_FORMAT = (
    "\n\n=== FORMATO DE RESPUESTA ===\n"
    "\n"
    "<Thinking>\n"
    "[Razonamiento campo por campo: qué evidencia encontraste\n"
    " y qué decisión tomaste.]\n"
    "</Thinking>\n"
    "\n"
    "Paso 1: Razona en <Thinking> ... </Thinking>.\n"
    "Paso 2: Genera el JSON de salida válido.\n"
    "El JSON debe cumplir EXACTAMENTE con el esquema objetivo.\n"
    "NO incluyas nada después del JSON."
)

# ═══════════════════════════════════════════════════════
#  Assembler
# ═══════════════════════════════════════════════════════


class PromptAssembler:
    """Build an adaptive prompt from a TaskProfile."""

    def __init__(self, budget: int = _BUDGET):
        self.budget = budget

    def assemble(
        self,
        profile: TaskProfile,
        instruction: str,
        input_text: str,
        schema_json: str,
    ) -> str:
        """Assemble the full prompt within budget.

        Args:
            profile: TaskProfile from SchemaAnalyzer.
            instruction: Task instruction.
            input_text: Source text.
            schema_json: Pretty-printed JSON schema.

        Returns:
            Complete prompt string.
        """
        # ── 1. Template (always) ──────────────────────────
        template = (
            "=== TAREA ===\n"
            f"{instruction}\n"
            "\n"
            "=== TEXTO ===\n"
            f"{input_text}\n"
            "\n"
            "=== ESQUEMA OBJETIVO ===\n"
            f"{schema_json}\n"
            "\n"
            "=== INSTRUCCIONES ===\n"
        )

        # ── 2. Select instruction modules ─────────────────
        modules: list[str] = []

        used = 0

        def _try_add(text: str) -> bool:
            """Add a module if within budget."""
            nonlocal used
            if used + len(text) <= self.budget:
                modules.append(text)
                used += len(text)
                return True
            return False

        # a) BASE (always)
        _try_add(BASE)

        # b) COMPLEXITY
        if profile.complexity_confidence >= _THRESHOLD_COMPLEXITY:
            ctext = COMPLEXITY_MODULES.get(
                profile.complexity, _COMPLEXITY_DEFAULT
            )
        else:
            ctext = _COMPLEXITY_DEFAULT
        _try_add(ctext)

        # c) FIELD_TYPES — based on current schema
        seen_numeric = False
        for ft in profile.field_types:
            if ft in ("number", "integer") and seen_numeric:
                continue  # deduplicate (both trigger same NUMÉRICOS rule)
            rule_text = FIELD_RULES.get(ft)
            if rule_text:
                _try_add(rule_text)
                if ft in ("number", "integer"):
                    seen_numeric = True
        if profile.has_nullable:
            _try_add(NULLABLE_RULE)

        # d) PATTERN_ERRORS
        if profile.has_known_errors and profile.closest_similarity >= _THRESHOLD_PATTERN:
            error_text = (
                "\n\n[ERRORES CONOCIDOS: " + profile.pattern + "]\n" +
                "\n".join(f"- {rule}" for rule in profile.error_rules)
            )
            _try_add(error_text)

        # e) DOMAIN_TIPS
        if profile.domain_confidence >= _THRESHOLD_DOMAIN and profile.domain:
            dtext = DOMAIN_TIPS.get(profile.domain, DOMAIN_DEFAULT)
            if dtext:
                _try_add(dtext)

        # f) FEW_SHOT (only if very similar)
        # (We skip this in the initial implementation — can be added later)

        # ── 3. Assemble ───────────────────────────────────
        instructions = "".join(modules)
        prompt = template + instructions + _OUTPUT_FORMAT

        return prompt


# ── Helpers ─────────────────────────────────────────────


def count_chars(prompt: str) -> int:
    """Count non-template characters (instruction + format)."""
    # For debugging: count everything after === INSTRUCCIONES ===
    marker = "=== INSTRUCCIONES ===\n"
    idx = prompt.find(marker)
    if idx == -1:
        return len(prompt)
    return len(prompt) - idx - len(marker)
