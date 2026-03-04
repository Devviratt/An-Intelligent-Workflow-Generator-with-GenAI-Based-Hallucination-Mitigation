import asyncio
import httpx
import traceback

async def test():
    async with httpx.AsyncClient(timeout=60) as client:
        prompt = "Hello, what is 2+2?"
        try:
            print("Testing Ollama generation...")
            print("Sending request to http://localhost:11434/api/generate")
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                }
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                resp = data.get("response", "")[:200]
                print(f"✓ Generation working! Response: {resp}...")
                return True
            else:
                print(f"✗ Error: {response.text[:500]}")
                return False
        except Exception as e:
            print(f"✗ Exception: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = asyncio.run(test())
    exit(0 if success else 1)

