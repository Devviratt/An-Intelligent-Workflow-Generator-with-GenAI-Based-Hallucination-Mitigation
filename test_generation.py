#!/usr/bin/env python3
"""Test workflow generation via REST API."""

import httpx
import json
import time

time.sleep(1)  # Wait for server to be fully ready

request_data = {
    "instruction": "online payment processing: customer initiates payment, system verifies payment, payment is processed, receipt is sent to customer",
    "mode": "workflow"
}

print(f"Testing workflow generation...\nRequest: {json.dumps(request_data)}\n")

try:
    response = httpx.post(
        "http://localhost:8000/api/v1/generate",
        json=request_data,
        timeout=180  # 3 minutes for complete request
    )
    
    result = response.json()
    
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Success: {result.get('success')}")
    
    workflow = result.get('workflow')
    if workflow:
        print(f"✓ Nodes: {len(workflow.get('nodes', []))}")
        print(f"✓ Edges: {len(workflow.get('edges', []))}")
    else:
        print("✗ No workflow generated")
        
    print(f"✓ Generation Time: {result.get('metrics', {}).get('generation_time_ms', 0):.1f}ms")
    
    if result.get('errors'):
        print(f"\n✗ Errors:")
        for err in result['errors']:
            print(f"  - {err.get('message', 'Unknown error')[:300]}")
    else:
        print("\n✓ No errors!")
        
    if workflow and workflow.get('nodes'):
        print(f"\nGenerated workflow nodes:")
        for node in workflow['nodes'][:5]:
            print(f"  - {node.get('label', 'unknown')}")
    
except Exception as e:
    import traceback
    print(f"✗ Error: {type(e).__name__}: {e}")
    traceback.print_exc()
