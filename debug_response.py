#!/usr/bin/env python3
"""Debug workflow generation response."""

import httpx
import json

request_data = {
    "instruction": "online payment processing: customer initiates payment, system verifies payment, payment is processed, receipt is sent to customer",
    "mode": "workflow"
}

print("Testing workflow generation with full output...\n")

try:
    response = httpx.post(
        "http://localhost:8000/api/v1/generate",
        json=request_data,
        timeout=180
    )
    
    result = response.json()
    print(json.dumps(result, indent=2)[:2000])  # Print first 2000 chars
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
