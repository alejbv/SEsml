import os
import json
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

        # Call OpenAI with the task's JSON schema
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
                    "schema": task.target_schema,
                    "strict": True,
                },
            },
        )
        self.usage.add(getattr(response, "usage", None))

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
