"""
Adaptive Pipeline — One-shot extraction with adaptive prompt assembly.

Pipeline:
  1. SchemaOptimizer (local, no LLM) → optimized schema
  2. SchemaAnalyzer (local) → TaskProfile
  3. PromptAssembler → adaptive prompt within budget
  4. Single LLM call (visible CoT + direct JSON)
  5. JSON extraction via ``_extract_json``

Key innovation: instead of a fixed prompt, the prompt is assembled
module-by-module based on the task's complexity, field types,
domain, and similarity to known tasks in data/dev.
"""

from __future__ import annotations

import json
import os
from logging import getLogger
from typing import Any
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from gensie.agent import GenSIEAgent
from gensie.task import Task
from .optimizer import SchemaOptimizer
from .schema_analyzer import SchemaAnalyzer
from .prompt_assembler import PromptAssembler
from .decoder import _extract_json

load_dotenv()
logger = getLogger("gensie")

# ── Default paths ───────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_DEFAULT_VECTORS = _HERE / "data" / "dev_vectors.json"
_DEFAULT_ERROR_MAP = _HERE / "data" / "error_map.json"


class AdaptivePipelineAgent(GenSIEAgent):
    """One-shot extraction with adaptive prompt assembly.

    Key difference from HybridCoT: the prompt is built dynamically
    based on the task profile (complexity, field types, domain,
    known error patterns).

    Uses a single LLM call without ``response_format``, allowing
    visible CoT (<Thinking> tags) + direct JSON output.
    """

    def __init__(
        self,
        pydantic_model_class: type | None = None,
        vectors_path: str | Path = _DEFAULT_VECTORS,
        error_map_path: str | Path = _DEFAULT_ERROR_MAP,
        budget: int = 1400,
    ):
        self.pydantic_model_class = pydantic_model_class
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY") or "sk-dummy",
        )
        self.analyzer = SchemaAnalyzer(
            vectors_path=vectors_path,
            error_map_path=error_map_path,
        )
        self.assembler = PromptAssembler(budget=budget)

    def run(self, task: Task, model: str) -> dict[str, Any]:
        """Execute the adaptive extraction pipeline.

        Args:
            task: Task with input_text, instruction, target_schema.
            model: Model name.

        Returns:
            Extracted JSON dict.
        """
        # ── Step 0: Schema Optimization ───────────────────
        logger.info("[adaptive] Step 0: Schema Optimization")
        raw_schema = task.target_schema
        optimized_schema = SchemaOptimizer.from_raw_schema(raw_schema)

        # ── Step 1: Schema Analysis ───────────────────────
        logger.info("[adaptive] Step 1: Schema Analysis")
        # Convert task to dict for the analyzer
        task_dict = {
            "target_schema": {
                "title": raw_schema.get("title", ""),
                "description": raw_schema.get("description", ""),
                "properties": raw_schema.get("properties", {}),
            },
            "instruction": task.instruction,
            "metadata": task.metadata if hasattr(task, "metadata") and task.metadata else {},
        }

        profile = self.analyzer.analyze(task_dict)

        logger.info(
            "[adaptive] Profile: complexity=%s (conf=%.2f), "
            "domain=%s (conf=%.2f), field_types=%s, "
            "pattern=%s, closest=%s (sim=%.3f)",
            profile.complexity,
            profile.complexity_confidence,
            profile.domain,
            profile.domain_confidence,
            profile.field_types,
            profile.pattern,
            profile.closest_task_id,
            profile.closest_similarity,
        )

        # ── Step 2: Prompt Assembly ───────────────────────
        logger.info("[adaptive] Step 2: Prompt Assembly")
        schema_json = json.dumps(optimized_schema, indent=2, ensure_ascii=False)
        prompt = self.assembler.assemble(
            profile=profile,
            instruction=task.instruction,
            input_text=task.input_text,
            schema_json=schema_json,
        )

        prompt_chars = len(prompt)
        logger.info(
            "[adaptive] Prompt assembled: %d chars total, %d instruction chars",
            prompt_chars,
            self._count_instruction_chars(prompt),
        )

        # ── Step 3: LLM Call ─────────────────────────────
        logger.info("[adaptive] Step 3: LLM Call (%s)", model)
        response = self._call_model(model, prompt)
        content = response.strip()
        if not content:
            raise ValueError("Empty response from model in Adaptive pipeline.")

        # ── Step 4: JSON Extraction ───────────────────────
        try:
            result = _extract_json(content)
        except ValueError as e:
            logger.error("[adaptive] JSON extraction failed: %s", e)
            logger.error("[adaptive] Response preview: %s...", content[:500])
            raise

        logger.info("[adaptive] Extraction completed successfully.")
        return result

    def _call_model(self, model: str, prompt: str, max_tokens: int = 4096) -> str:
        """Call LLM WITHOUT response_format for visible CoT + JSON."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("[adaptive] LLM call failed: %s", e)
            raise
        return response.choices[0].message.content or ""

    @staticmethod
    def _count_instruction_chars(prompt: str) -> int:
        """Count only the instruction part of the prompt."""
        marker = "=== INSTRUCCIONES ===\n"
        idx = prompt.find(marker)
        if idx == -1:
            return 0
        return len(prompt) - idx - len(marker)
