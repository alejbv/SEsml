"""
HybridCoT - One-shot extraction with visible CoT + direct JSON.

Pipeline:
  1. SchemaOptimizer (local, no LLM).
  2. Single LLM call with visible <Thinking> reasoning + direct JSON.
  3. JSON extraction via _extract_json.

Middleware between ExtractionAgent (2 calls) and SingleCallAgent (internal CoT).
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
from .prompts import HYBRID_COT_PROMPT
from .decoder import _extract_json

load_dotenv()
logger = getLogger("gensie")


class HybridCoTAgent(GenSIEAgent):
    """One-shot extraction: visible CoT + direct JSON, no response_format."""

    def __init__(self, pydantic_model_class=None):
        self.pydantic_model_class = pydantic_model_class
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY") or "sk-dummy",
        )

    def run(self, task: Task, model: str) -> dict[str, Any]:
        raw_schema = task.target_schema
        logger.info("[hybrid_cot] Step 0: Schema Optimization")
        optimized_schema = SchemaOptimizer.from_raw_schema(raw_schema)

        logger.info("[hybrid_cot] Step 1: Single LLM call (visible CoT)")
        schema_json = json.dumps(optimized_schema, indent=2, ensure_ascii=False)
        prompt = HYBRID_COT_PROMPT.format(
            instruction=task.instruction,
            input_text=task.input_text,
            schema_json=schema_json,
        )

        response = self._call_model(model, prompt)
        content = response.strip()
        if not content:
            raise ValueError("Empty response from model in HybridCoT pipeline.")

        try:
            result = _extract_json(content)
        except ValueError as e:
            logger.error("[hybrid_cot] JSON extraction failed: %s", e)
            logger.error("[hybrid_cot] Response preview: %s...", content[:500])
            raise

        logger.info("[hybrid_cot] Extraction completed successfully.")
        return result

    def _call_model(self, model: str, prompt: str, max_tokens: int = 4096) -> str:
        """Call LLM WITHOUT response_format so model can output thinking + JSON."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("[hybrid_cot] LLM call failed: %s", e)
            raise
        return response.choices[0].message.content or ""
