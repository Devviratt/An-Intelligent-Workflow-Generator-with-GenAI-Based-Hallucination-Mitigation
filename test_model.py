#!/usr/bin/env python3
"""Test Ollama phi3:mini model."""

import httpx

print("Testing phi3:mini model...")

try:
    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3:mini",
            "prompt": "What is 2+2?",
            "stream": False
        },
        timeout=60
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result.get('response', 'N/A')[:300]}")
        print("✓ Model is working!")
    else:
        print(f"Error: {response.text[:500]}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
