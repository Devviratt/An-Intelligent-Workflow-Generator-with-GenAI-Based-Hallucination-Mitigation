#!/usr/bin/env python3
"""
LLM Setup and Testing Script

Quick start script to:
1. Check Ollama connectivity
2. List available models
3. Generate fine-tuning data
4. Test LLM workflow generation

Usage:
    python scripts/setup_llm.py --generate-training
    python scripts/setup_llm.py --test-generation
    python scripts/setup_llm.py --check-ollama
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and list available models."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                logger.error("Ollama returned status %s", response.status_code)
                return False

            data = response.json()
            models = data.get("models", [])

            print(f"\n✓ Ollama is running at {base_url}")
            print(f"\nAvailable models ({len(models)}):")
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024**3)
                print(f"  - {name} ({size_gb:.1f} GB)")

            if not models:
                print("  (No models found. Run: ollama pull mistral)")
            return True

    except httpx.ConnectError:
        logger.error(
            "Cannot connect to Ollama at %s\n"
            "Make sure Ollama is running: ollama serve",
            base_url,
        )
        return False
    except Exception as exc:
        logger.error("Error checking Ollama: %s", exc)
        return False


async def test_generation(
    base_url: str = "http://localhost:11434",
    model: str = "mistral",
) -> bool:
    """Test LLM workflow generation."""
    try:
        logger.info("Testing LLM workflow generation...")

        prompt = """You are a workflow generation expert. Generate a simple 3-4 step workflow JSON.

Return ONLY valid JSON:
{
  "nodes": [
    {"id": "start", "label": "Start", "type": "start", "domain_step_id": "start"},
    {"id": "process", "label": "Process", "type": "process", "domain_step_id": "process"},
    {"id": "end", "label": "End", "type": "end", "domain_step_id": "end"}
  ],
  "edges": [
    {"source": "start", "target": "process", "condition": null},
    {"source": "process", "target": "end", "condition": null}
  ]
}

Generate now:"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    },
                },
            )

            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "")

            print(f"\n✓ Generated response (first 200 chars):")
            print(f"  {response_text[:200]}...")

            # Try to parse JSON
            import re
            match = re.search(r"\{[\s\S]*\}", response_text)
            if match:
                json_str = match.group()
                workflow = json.loads(json_str)
                print(f"\n✓ Valid JSON parsed:")
                print(f"  Nodes: {len(workflow.get('nodes', []))}")
                print(f"  Edges: {len(workflow.get('edges', []))}")
                return True
            else:
                print("\n✗ Could not parse JSON from response")
                return False

    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama")
        return False
    except Exception as exc:
        logger.error("Test failed: %s", exc)
        return False


def generate_training_data(output_dir: str = "data/training") -> bool:
    """Generate fine-tuning data from datasets."""
    try:
        from src.engines.domain_engine import DomainDatasetEngine
        from src.engines.finetuning import FineTuneDataGenerator

        logger.info("Loading datasets...")
        engine = DomainDatasetEngine()
        engine.load_all_sync()
        datasets = engine.all_datasets()

        logger.info(f"Loaded {len(datasets)} domains")

        logger.info("Generating training data...")
        generator = FineTuneDataGenerator()
        generator.generate_for_all_datasets(
            datasets,
            output_dir=Path(output_dir),
        )

        output_file = Path(output_dir) / "all_domains_training.jsonl"
        if output_file.exists():
            lines = output_file.read_text().strip().split("\n")
            print(f"\n✓ Training data saved to: {output_file}")
            print(f"  Total examples: {len(lines)}")
            return True
        else:
            print("\n✗ Failed to generate training data")
            return False

    except Exception as exc:
        logger.error("Training data generation failed: %s", exc)
        return False


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Setup and Testing Script",
    )
    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Check Ollama connectivity and list models",
    )
    parser.add_argument(
        "--test-generation",
        action="store_true",
        help="Test LLM workflow generation",
    )
    parser.add_argument(
        "--generate-training",
        action="store_true",
        help="Generate fine-tuning data from datasets",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Model name in Ollama",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks and generation",
    )

    args = parser.parse_args()

    # Default to checking Ollama if no specific action given
    if not any([args.check_ollama, args.test_generation,
                args.generate_training, args.all]):
        args.check_ollama = True

    print("\n" + "=" * 70)
    print("LLM Setup and Testing")
    print("=" * 70)

    success = True

    # Check Ollama
    if args.check_ollama or args.all:
        print("\n[1/3] Checking Ollama...")
        if not await check_ollama(args.base_url):
            success = False

    # Test generation
    if args.test_generation or args.all:
        print("\n[2/3] Testing Generation...")
        if not await test_generation(args.base_url, args.model):
            success = False

    # Generate training data
    if args.generate_training or args.all:
        print("\n[3/3] Generating Training Data...")
        if not generate_training_data(args.output_dir):
            success = False

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("✓ All checks passed! LLM setup is ready.")
        print("\nNext steps:")
        print("1. Start the server: uvicorn src.api.server:app --reload")
        print("2. Access playground: http://localhost:5173")
        print("3. Generate workflows via UI or API")
        sys.exit(0)
    else:
        print("✗ Some checks failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
