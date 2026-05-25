"""
SingleCall — One-shot extraction with constrained decoding + internal CoT.

Pipeline:
  1. SchemaOptimizer — resolves $ref, cleans schema (local, no LLM).
  2. Single LLM call — merged prompt (instruction + text + schema)
     with ``response_format=json_schema`` for guaranteed valid JSON output.
     - Model reasons internally before generating the JSON.
     - No visible CoT in output — constrained decoding ensures valid JSON.
  3. JSON extraction — parses response (``_extract_json`` safety net).

This replaces the 2-call Draft + Decoder pipeline, reducing total latency
by eliminating the intermediate draft step. Internal CoT still guides
extraction quality, but all reasoning happens inside one LLM call.

Usage:
    >>> from gensie.pipelines.extraction import SingleCallAgent
    >>> agent = SingleCallAgent()
    >>> result = agent.run(task, model="gpt-4o")
"""

from __future__ import annotations

import json
import os
from logging import getLogger
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from gensie.agent import GenSIEAgent
from gensie.task import Task
from .optimizer import SchemaOptimizer
from .prompts import SINGLE_CALL_PROMPT
from .decoder import _extract_json

load_dotenv()
logger = getLogger("gensie")


class SingleCallAgent(GenSIEAgent):
    """One-shot extraction agent: constrained decoding + internal CoT.

    Uses SchemaOptimizer (local) + 1 LLM call with ``response_format=json_schema``.
    The model reasons internally and outputs valid JSON directly —
    no intermediate draft, no visible CoT tags.
    """

    def __init__(self, pydantic_model_class: type | None = None):
        self.pydantic_model_class = pydantic_model_class
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY") or "sk-dummy",
        )

    def run(self, task: Task, model: str) -> dict[str, Any]:
        """Execute the single-call extraction pipeline.

        Args:
            task: The task with input_text, instruction, target_schema.
            model: Model name (provided by the evaluator).

        Returns:
            Dict with the extracted JSON.
        """
        raw_schema = task.target_schema
        input_text = task.input_text

        # -- Step 0: Schema Optimization (local, no LLM) --
        logger.info("[single_call] Step 0: Schema Optimization")
        optimized_schema = SchemaOptimizer.from_raw_schema(raw_schema)

        # -- Step 1: Single LLM call with constrained decoding --
        logger.info("[single_call] Step 1: Single LLM call (json_schema)")

        schema_json = json.dumps(optimized_schema, indent=2, ensure_ascii=False)
        prompt = SINGLE_CALL_PROMPT.format(
            instruction=task.instruction,
            input_text=input_text,
            schema_json=schema_json,
        )

        response = self._call_model(model, prompt, schema=optimized_schema)

        # -- Step 2: Parse JSON from response --
        content = response.strip()
        if not content:
            raise ValueError("Empty response from model in SingleCall pipeline.")

        try:
            result = _extract_json(content)
        except ValueError as e:
            logger.error("[single_call] JSON extraction failed: %s", e)
            logger.error("[single_call] Response preview: %s...", content[:500])
            raise

        logger.info("[single_call] Extraction completed successfully.")
        return result

    # -- LLM call with constrained decoding --

    def _call_model(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int = 4096,
    ) -> str:
        """Call the LLM with ``response_format=json_schema`` for valid JSON output."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": schema,
                    },
                },
            )
        except Exception as e:
            logger.error("[single_call] LLM call failed: %s", e)
            raise

        return response.choices[0].message.content or ""
