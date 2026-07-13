#!/usr/bin/env python
"""Quick test of hybrid_cot and adaptive pipelines."""
import json, sys, os
sys.path.insert(0, '/home/alejbv/Projects/research/gensie/src')
os.environ['OPENAI_BASE_URL'] = 'http://10.6.125.216:8080/v1'
os.environ['OPENAI_API_KEY'] = 'sk-dummy'

from gensie.baseline import OfficialParticipant
from gensie.task import Task

p = OfficialParticipant()
print("Registered pipelines:", list(p.pipelines.keys()))

task = Task.load('/home/alejbv/Projects/research/gensie/data/starter/cultural_media_001.json')
print(f"\nTask: {task.id}")
print(f"Schema: {task.target_schema.get('title', 'unknown')}")

# Test hybrid_cot
print("\n--- Testing hybrid_cot ---")
try:
    agent = p.get_agent('hybrid_cot')
    print(f"Agent type: {type(agent).__name__}")
    result = agent.run(task, model='qwen3-14b')
    if isinstance(result, dict):
        print(f"Result keys: {list(result.keys())[:5]}...")
        print(f"Result (truncated): {json.dumps(result, indent=2)[:500]}")
    else:
        print(f"Result (not dict): {result}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test adaptive
print("\n--- Testing adaptive ---")
try:
    agent = p.get_agent('adaptive')
    print(f"Agent type: {type(agent).__name__}")
    result = agent.run(task, model='qwen3-14b')
    if isinstance(result, dict):
        print(f"Result keys: {list(result.keys())[:5]}...")
        print(f"Result (truncated): {json.dumps(result, indent=2)[:500]}")
    else:
        print(f"Result (not dict): {result}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Compare with baseline output
print("\n--- Testing baseline (for comparison) ---")
try:
    agent = p.get_agent('baseline')
    result = agent.run(task, model='qwen3-14b')
    if isinstance(result, dict):
        print(f"Baseline keys: {list(result.keys())[:5]}...")
        # Check if output is populated
        print(f"Output fields: {len(result)} top-level keys")
    else:
        print(f"Baseline result: {result}")
except Exception as e:
    print(f"Baseline error: {e}")
