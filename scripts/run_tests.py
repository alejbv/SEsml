"""Quick sanity tests for extraction pipeline imports and schema analysis."""
import sys
import json
import os

sys.path.insert(0, "src")

# 1. Import extraction modules
from gensie.extraction import ExtractionAgent, HybridCoTAgent, AdaptivePipelineAgent, SchemaOptimizer
print("1/6 ✅ Extraction modules imported")

# 2. Import baseline
from gensie.baseline import OfficialParticipant, BasicAgent
print("2/6 ✅ Baseline modules imported")

# 3. Create participant
participant = OfficialParticipant()
info = participant.get_info()
print(f"3/6 ✅ Participant: {info.team_name} ({info.institution})")
for p in info.pipelines:
    print(f"    Pipeline: {p.name}")

# 4. SchemaAnalyzer loads vectors
from gensie.extraction.schema_analyzer import SchemaAnalyzer
analyzer = SchemaAnalyzer()
print(f"4/6 ✅ SchemaAnalyzer: {len(analyzer._dev_vectors)} vectors loaded")

# 5. PromptAssembler works
from gensie.extraction.prompt_assembler import PromptAssembler
assembler = PromptAssembler()
print(f"5/6 ✅ PromptAssembler (budget={assembler.budget} chars)")

# 6. Analyze a real task
dev_dir = "data/dev"
dev_files = sorted(os.listdir(dev_dir))
with open(os.path.join(dev_dir, dev_files[0])) as f:
    sample = json.load(f)

task_dict = {
    "target_schema": {
        "title": sample["target_schema"]["title"],
        "description": sample["target_schema"]["description"],
        "properties": sample["target_schema"]["properties"],
    },
    "instruction": sample["instruction"],
    "metadata": sample.get("metadata", {}),
}
profile = analyzer.analyze(task_dict)
print(f"6/6 ✅ Profile: {profile.pattern} ({profile.complexity}, domain={profile.domain})")
print(f"    Similarity: {profile.closest_similarity:.3f} ({profile.closest_task_id})")
print(f"    Known errors: {profile.has_known_errors} ({len(profile.error_rules)} rules)")
print(f"    Field types: {profile.field_types}")

print("\n🎉 All 6 tests passed!")
