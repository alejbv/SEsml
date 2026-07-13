import os
import json
import copy
from typing import Any, Dict
from openai import OpenAI
from gensie.agent import GenSIEAgent, Participant, ParticipantInfo, PipelineInfo
from gensie.extraction import ExtractionAgent, HybridCoTAgent, AdaptivePipelineAgent
from gensie.task import Task
from gensie.usage import UsageTracker
from dotenv import load_dotenv
from logging import getLogger

load_dotenv()
logger = getLogger("gensie")

# Keys that LM Studio / constrained-decoding engines reject in JSON Schema.
_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "default", "title", "format",
    "minimum", "maximum", "minLength", "maxLength",
    "pattern",  # <-- regex patterns (esp. ?: non-capturing groups) cause 400 errors
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
})


def _strip_unsupported_schema_keys(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip keys that LM Studio's json_schema engine rejects.

    Operates on a **deep copy** so the original schema is never mutated.
    """
    result: dict[str, Any] = {}
    for k, v in schema.items():
        if k in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            # Property names are user-defined field names (e.g. "title",
            # "default", "format") — never strip them. Recurse into values.
            result[k] = {
                prop_name: (
                    _strip_unsupported_schema_keys(prop_schema)
                    if isinstance(prop_schema, dict)
                    else prop_schema
                )
                for prop_name, prop_schema in v.items()
            }
        elif isinstance(v, dict):
            result[k] = _strip_unsupported_schema_keys(v)
        elif isinstance(v, list):
            result[k] = [
                _strip_unsupported_schema_keys(i) if isinstance(i, dict) else i
                for i in v
            ]
        else:
            result[k] = v
    return result


class BasicAgent(GenSIEAgent):
    """
    Reference implementation using OpenAI Structured Outputs.
    Configurable via environment variables:
    - OPENAI_BASE_URL: (Optional) Custom endpoint for local LLMs.
    - OPENAI_API_KEY: (Required) Your API key.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY") or "sk-dummy",
        )
        # Tallies token usage for the current task; the server reads it to set
        # the X-GenSIE-Token-Usage response header. Reuse this in your own agent.
        self.usage = UsageTracker()

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        """
        Executes the extraction using OpenAI's response_format for strict schema compliance.
        """
        self.usage.reset()
        prompt = task.get_input_prompt()

        # Sanitize the target schema: strip keys that LM Studio rejects
        # (pattern with unsupported regex syntax, format, min/max, etc.)
        sanitized_schema = _strip_unsupported_schema_keys(
            copy.deepcopy(task.target_schema)
        )

        # Call OpenAI with the task's JSON schema
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data extraction agent.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": sanitized_schema,
                    },
                },
            )
            self.usage.add(getattr(response, "usage", None))
        except Exception as e:
            logger.error("[baseline] LLM call failed: %s", e)
            return {"error": f"LLM call failed: {str(e)}"}

        # Parse the structured JSON response
        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            # Fallback for unexpected API errors
            return {"error": f"Failed to parse model response: {str(e)}"}
        except Exception as e:
            logger.error(str(e))
            return {"error": str(e)}


class OfficialParticipant(Participant):
    """
    Standard entry point for the competition.
    Participants can configure up to 3 pipelines here.
    """

    def __init__(self):
        # Default pipeline using the reference BasicAgent
        self.pipelines = {
            "baseline": BasicAgent(),
            "extraction": ExtractionAgent(),
            "hybrid_cot": HybridCoTAgent(),
            "adaptive": AdaptivePipelineAgent(),
        }

    def get_info(self) -> ParticipantInfo:
        return ParticipantInfo(
            team_name="SEsml — Schema-guided Extraction with Small Language Models",
            institution="Universidad de La Habana",
            pipelines=[
                PipelineInfo(
                    name="baseline",
                    description=(
                        "BasicAgent with direct response_format=json_schema "
                        "(OpenAI Structured Outputs). No draft-then-decode, no "
                        "schema optimization, no CoT. Reference baseline for "
                        "gap-closed ranking."
                    ),
                ),

                PipelineInfo(
                    name="extraction",
                    description=(
                        "SchemaOptimizer + DraftEngine (borrador campo:valor) + "
                        "Constrained Decoding (2 fases). Prompt v3b con 3 reglas "
                        "críticas para SLMs. Pipeline principal para evaluación."
                    ),
                ),
                PipelineInfo(
                    name="hybrid_cot",
                    description=(
                        "SchemaOptimizer + llamada única con CoT visible "
                        "(<Thinking>) + JSON directo vía _extract_json. "
                        "Alternativa ligera al pipeline de 2 fases."
                    ),
                ),
                PipelineInfo(
                    name="adaptive",
                    description=(
                        "SchemaAnalyzer + PromptAssembler + llamada única. "
                        "Prompt adaptativo mediante similitud semántica sobre "
                        "vectores de tareas de desarrollo (data/dev_vectors.json)."
                    ),
                ),
            ],
        )

    def get_agent(self, pipeline_name: str) -> GenSIEAgent:
        if pipeline_name not in self.pipelines:
            # Fallback to extraction if pipeline not found
            return self.pipelines["extraction"]
        return self.pipelines[pipeline_name]
