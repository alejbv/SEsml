#!/usr/bin/env python3
"""Check what models are available on the inference server."""
import json, urllib.request, urllib.error, sys

BASE = "http://10.6.125.216:8080"

# Try listing models
try:
    req = urllib.request.Request(f"{BASE}/v1/models")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
        print("=== Available Models ===")
        print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Models list failed: {e}")

# Try a quick chat completion with common model names
models_to_try = [
    "qwen3-14b",
    "Qwen3-14B",
    "qwen3-14b-instruct",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-Instruct",
    "gemma-4b",
    "Gemma-4B",
    "google/gemma-4b",
    "google/gemma-4b-it",
    "gemma-4b-it",
    "meta-llama/llama-3.2-3b-instruct",
    "llama-3.2-3b",
]

print("\n=== Testing model names ===")
for model in models_to_try:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 5
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        method="POST"
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read())
            content = resp["choices"][0]["message"]["content"]
            print(f"  ✅ {model}: responds! \"{content}\"")
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        if "model_not_found" in str(err) or "not found" in str(err).lower():
            print(f"  ❌ {model}: not found")
        else:
            print(f"  ⚠️  {model}: {err.get('error', {}).get('message', str(e)[:100])}")
    except Exception as e:
        print(f"  ❌ {model}: {type(e).__name__}: {str(e)[:80]}")

print("\nDone.")
