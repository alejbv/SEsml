"""
DraftEngine — Generates a semi-structured draft via a single LLM call.

The model produces <Thinking>…</Thinking><Draft>campo: valor</Draft>.

The draft is NOT JSON — it is semi-structured text (``campo: valor`` lines).
Downstream ``decode_to_json`` (constrained decoding) handles the final JSON.

Returns a **single string** — the content to pass to the decoder.
If nothing usable is extracted, returns ``None`` (caller must handle).
"""


import json
from logging import getLogger
from typing import Any

from .prompts import DRAFT_PROMPT, parse_draft_response

logger = getLogger("gensie")


class DraftEngine:
    """Generates a semi-structured draft from text + schema + instruction.

    Makes exactly **one** LLM call. Returns the raw draft text.

    Usage::

        draft_engine = DraftEngine(client, model)
        draft = draft_engine.generate(schema, text, instruction)
        if draft is None:
            # handle error — no draft could be extracted
    """

    def __init__(
        self,
        client: Any,
        model: str,
    ):
        self._client = client
        self._model = model

    def generate(
        self,
        optimized_schema: dict[str, Any],
        input_text: str,
        instruction: str,
        *,
        max_tokens: int = 2048,
    ) -> str | None:
        """Run the single-call draft generation.

        Args:
            optimized_schema: Schema dict from SchemaOptimizer.optimized_schema.
            input_text: Raw input text to extract from.
            instruction: The natural language task description
                (e.g., "Analiza la descripción del software...").
            max_tokens: Max tokens for the draft output.

        Returns:
            Semi-structured draft text (``campo: valor`` lines), or ``None``
            if the model response contained no usable content.
        """
        schema_json = json.dumps(optimized_schema, indent=2, ensure_ascii=False)

        prompt = DRAFT_PROMPT.format(
            instruction=instruction,
            input_text=input_text,
            schema_json=schema_json,
        )

        response = self._call_model(prompt, max_tokens=max_tokens)

        content = parse_draft_response(response)

        if content is None:
            # Fallback: si el modelo respondió sin tags XML,
            # mandamos la respuesta completa al decoder.
            # El decoder tiene 4 estrategias de parseo (JSON,
            # markdown, {…}, KV lines) y puede extraer algo útil.
            stripped = response.strip()
            if stripped:
                logger.warning(
                    "[draft] No se encontraron tags <Draft>/<Thinking>. "
                    "Se envía respuesta raw al decoder (%d caracteres).",
                    len(stripped),
                )
                return stripped

            logger.error(
                "[draft] No se pudo extraer draft ni thinking. "
                "Respuesta vacía."
            )
            return None

        logger.info("[draft] Draft extraído (%d caracteres)", len(content))
        return content

    # ── LLM call ────────────────────────────────────────────────────────

    def _call_model(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call the LLM via OpenAI-compatible API."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"[draft] LLM call failed: {e}")
            raise

        return response.choices[0].message.content or ""
