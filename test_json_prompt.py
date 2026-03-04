#!/usr/bin/env python3
"""Test phi3:mini with simple JSON prompt."""

import httpx

prompt = """Generate a JSON workflow with 3 nodes:
{"nodes": [{"id": "1", "label": "Start", "type": "start"}]}"""

print("Testing phi3:mini with JSON prompt...")
print(f"Prompt length: {len(prompt)} characters\n")

try:
    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 256,  # Very short for testing
            },
        },
        timeout=90
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        resp_text = result.get('response', '')[:500]
        print(f"Response:\n{resp_text}\n")
        print("✓ Success!")
    else:
        print(f"Error: {response.text[:500]}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
