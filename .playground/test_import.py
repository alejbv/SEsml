"""Test that gensie modules import correctly."""
import sys
print(f"Python {sys.version}")

from gensie.cli import app
print("CLI module: OK")

from gensie.server import app as fastapi_app
print("Server module: OK")

from gensie.agent import GenSIEAgent, Participant
print("Agent module: OK")

from gensie.baseline import OfficialParticipant, BasicAgent
print("Baseline module: OK")

from gensie.extraction import ExtractionAgent, HybridCoTAgent, AdaptivePipelineAgent
print("Extraction modules: OK")

print("\nAll imports successful!")
