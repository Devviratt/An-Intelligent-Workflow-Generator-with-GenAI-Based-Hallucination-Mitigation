#!/usr/bin/env python
import asyncio
import httpx
from src.config import settings

async def test_ollama_response():
    """Test what Ollama actually generates."""
    
    context = """Domain: Online Payment Processing
Description: End-to-end payment processing workflow including validation, fraud detection, authorization, and settlement.

Available steps:
- initiate_payment: User Initiates Payment [REQUIRED] (start)
- validate_input: Validate Payment Input [REQUIRED] (process)
- validate_card: Validate Card Details [REQUIRED] (process)
- check_balance: Check Account Balance [optional] (process)
- fraud_check: Fraud Detection Check [REQUIRED] (decision)
- manual_review: Manual Fraud Review [optional] (process)
- bank_authorization: Bank Authorization [REQUIRED] (decision)
- apply_discount: Apply Discount / Coupon [optional] (process)
- process_settlement: Process Settlement [REQUIRED] (process)
- generate_receipt: Generate Receipt [REQUIRED] (process)
- send_notification: Send Notification [optional] (process)
- transaction_complete: Transaction Complete [REQUIRED] (end)
- transaction_failed: Transaction Failed [optional] (end)
- retry_payment: Retry Payment [optional] (process)

Must include steps: initiate_payment, validate_input, validate_card, fraud_check, bank_authorization, process_settlement, generate_receipt, transaction_complete"""

    prompt = f"""{context}

User instruction: "Generate a payment processing workflow with fraud detection"

Generate a valid JSON workflow. Output ONLY the JSON object, no other text.

JSON format (REQUIRED):
{{"nodes": [{{"id": "start", "label": "Start", "type": "start", "domain_step_id": ""}}], "edges": []}}

JSON:
"""

    print("Sending to Ollama...")
    print(f"Prompt length: {len(prompt)} characters")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 512,
                    },
                },
            )
            
            result = response.json()
            generated_text = result.get("response", "")
            
            print(f"Status: {response.status_code}")
            print(f"Generated text ({len(generated_text)} chars):")
            print("=" * 60)
            print(generated_text)
            print("=" * 60)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama_response())
