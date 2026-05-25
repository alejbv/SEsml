"""
Extraction  — Orchestrator.

Pipeline:
  1. SchemaOptimizer → optimized_schema
  2. DraftEngine → semi-structured draft (1 LLM call, ``campo: valor`` lines)
  3. Constrained Decoding → final JSON (1 LLM call, Draft-as-Context)
     └── Fallback: local semi-structured parsing from draft if CD fails
"""


import os
from logging import getLogger
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from gensie.agent import GenSIEAgent
from gensie.task import Task
from .optimizer import SchemaOptimizer

from .draft import DraftEngine
from .decoder import decode_to_json

load_dotenv()
logger = getLogger("gensie")


class ExtractionAgent(GenSIEAgent):
    """Extraction v2 pipeline agent.

    Uses SchemaOptimizer + DraftEngine (1 call, semi-structured draft) +
    Constrained Decoding (Draft-as-Context).
    """

    def __init__(self, pydantic_model_class: type | None = None):
        self.pydantic_model_class = pydantic_model_class
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY") or "sk-dummy",
        )

    def run(self, task: Task, model: str) -> dict[str, Any]:
        """Execute the full extraction pipeline for a Task.

        Pipeline: SchemaOptimizer → DraftEngine (1 call) → Constrained Decoding.

        Args:
            task: The task with input_text, instruction, target_schema.
            model: Model name (provided by the evaluator).

        Returns:
            Dict with the extracted JSON.
        """
        raw_schema = task.target_schema
        input_text = task.input_text

        # ── FASE 0: Schema Optimization ────────────────────────────────
        logger.info("[extraction] Fase 0: Schema Optimization")
        optimized_schema = SchemaOptimizer.from_raw_schema(raw_schema)

        # ── FASE 1: Draft Generation (single merged call) ──────────────
        logger.info("[extraction] Fase 1: Draft Generation")
        draft_engine = DraftEngine(client=self.client, model=model)
        draft = draft_engine.generate(
            optimized_schema=optimized_schema,
            input_text=input_text,
            instruction=task.instruction,
        )

        if draft is None:
            logger.error(
                "[extraction] DraftEngine no produjo contenido. "
                "Se procede con decoder sin borrador."
            )

        # ── FASE 2: Constrained Decoding ───────────────────────────────
        logger.info("[extraction] Fase 2: Constrained Decoding")
        result_json = decode_to_json(
            client=self.client,
            model=model,
            draft=draft,          # puede ser None → decoder lo maneja
            schema=optimized_schema,
        )

        return result_json
