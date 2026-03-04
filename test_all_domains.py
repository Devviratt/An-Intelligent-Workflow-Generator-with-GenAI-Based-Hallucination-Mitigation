#!/usr/bin/env python3
"""Comprehensive test of the LLM-based workflow generation system."""

import httpx
import json
import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_all_domains():
    """Test workflow generation across all domains."""
    test_cases = [
        ("online_payment", "process an online payment: customer enters card details, system validates, payment confirmed, receipt sent"),
        ("user_registration", "register a new user: accept credentials, validate email, create account, send confirmation"),
        ("order_fulfillment", "complete an order: receive order, check inventory, pack items, ship to customer"),
        ("incident_response", "respond to a security incident: detect breach, isolate system, investigate, remediate, notify"),
        ("data_pipeline", "build a data pipeline: extract data from source, transform and clean, load to warehouse, generate report"),
        ("ci_cd_deployment", "deploy an application: commit code, run tests, build Docker image, deploy to production"),
    ]
    
    results = []
    for domain, instruction in test_cases:
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"Instruction: {instruction[:80]}...")
        print('='*60)
        
        try:
            response = httpx.post(
                "http://localhost:8000/api/v1/generate",
                json={
                    "instruction": instruction,
                    "mode": "workflow"
                },
                timeout=180
            )
            
            result = response.json()
            success = result.get('success', False)
            nodes_count = len(result.get('workflow', {}).get('nodes', [])) if result.get('workflow') else 0
            edges_count = len(result.get('workflow', {}).get('edges', [])) if result.get('workflow') else 0
            gen_time = result.get('metrics', {}).get('generation_time_ms', 0)
            
            print(f"[OK] Status: {response.status_code}")
            print(f"[OK] Success: {success}")
            print(f"[OK] Nodes: {nodes_count}")
            print(f"[OK] Edges: {edges_count}")
            print(f"[OK] Time: {gen_time/1000:.1f}s")
            
            if result.get('errors'):
                print(f"[ERROR] {result['errors'][0]['message'][:150]}")
                results.append((domain, False, 0, 0))
            else:
                print("[OK] No errors!")
                if result.get('workflow') and result['workflow'].get('nodes'):
                    print(f"  Nodes: {', '.join([n.get('label', '?')[:20] for n in result['workflow']['nodes'][:3]])}")
                results.append((domain, success, nodes_count, edges_count))
                
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {str(e)[:200]}")
            results.append((domain, False, 0, 0))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for domain, success, nodes, edges in results:
        status = "OK" if success else "FAIL"
        print(f"[{status}] {domain:20s} - Nodes: {nodes:2d}, Edges: {edges:2d}")
    
    successful = sum(1 for _, s, _, _ in results if s)
    print(f"\nTotal: {successful}/{len(results)} domains successful")

if __name__ == "__main__":
    test_all_domains()
