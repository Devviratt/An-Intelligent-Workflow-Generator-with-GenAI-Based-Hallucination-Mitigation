# LLM-Based Workflow Generation Setup Guide

## Overview

The system now uses **Llama (via Ollama)** as a local LLM to generate workflows instead of the deterministic dataset-driven approach. This enables more flexible, natural workflow generation with fine-tuned models.

## Architecture

```
User Instruction
       ↓
Instruction Parser (TF-IDF + keyword extraction)
       ↓
RAG Engine (builds domain context from datasets)
       ↓
LLM Generator (Ollama) → JSON workflow/flowchart
       ↓
Layout Engine (BFS-based positioning)
       ↓
GeneratedWorkflow output
```

## Prerequisites

### 1. Install Ollama

Download and install Ollama from: https://ollama.ai

### 2. Pull a Model

```bash
ollama pull mistral  # or your preferred model
ollama pull llama2
ollama pull neural-chat
```

### 3. Start Ollama Server

```bash
ollama serve
# By default listens on http://localhost:11434
```

## Configuration

Update `src/config.py`:

```python
# Local model settings
ollama_base_url: str = "http://localhost:11434"
ollama_model: str = "mistral"  # or your chosen model
ollama_timeout: float = 30.0
use_local_model: bool = True
```

Or set environment variables:

```bash
export WFG_OLLAMA_BASE_URL="http://localhost:11434"
export WFG_OLLAMA_MODEL="mistral"
export WFG_USE_LOCAL_MODEL="true"
```

## Fine-Tuning the LLM

### Option 1: Generate Fine-Tuning Data

```python
from pathlib import Path
from src.engines.domain_engine import DomainDatasetEngine
from src.engines.finetuning import FineTuneDataGenerator

# Load all datasets
engine = DomainDatasetEngine()
engine.load_all_sync()
datasets = engine.all_datasets()

# Generate JSONL training data
generator = FineTuneDataGenerator()
generator.generate_for_all_datasets(
    datasets,
    output_dir=Path("data/training"),
)
```

This creates `data/training/all_domains_training.jsonl` with instruction-workflow pairs.

### Option 2: Fine-Tune with Ollama

Once you have the JSONL file:

```bash
ollama create my-workflow-llm -f Modelfile
```

Create a `Modelfile`:

```dockerfile
FROM mistral
ADAPTER /path/to/finetuned-adapter

# Optional: set parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
```

Or use Ollama's fine-tuning API (if available):

```python
import httpx

async def finetune_model():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/finetune",
            json={
                "model": "mistral",
                "training_data": "data/training/all_domains_training.jsonl",
                "output_model": "my-workflow-llm",
            },
        )
```

## Usage

### Generate Workflow (API)

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Create an online payment processing workflow",
    "mode": "workflow",
    "domain_hint": "online_payment"
  }'
```

### Generate Flowchart (API)

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Build a payment flowchart with decision branches",
    "mode": "flowchart",
    "domain_hint": "online_payment"
  }'
```

### Python API

```python
from src.pipeline import Pipeline
from src.models.request import GenerateRequest, GenerationMode

async def main():
    pipeline = Pipeline()
    await pipeline.initialise()

    request = GenerateRequest(
        instruction="Generate a user registration workflow",
        mode=GenerationMode.WORKFLOW,
        domain_hint="user_registration",
    )

    response = await pipeline.generate(request)
    print(response.workflow.nodes)
    print(response.workflow.edges)
```

## Context (RAG) System

The **RAG Engine** automatically provides the LLM with:

1. **Domain metadata**: Description, compliance, SLA
2. **Step catalog**: All available steps with types and descriptions
3. **Allowed transitions**: Valid step-to-step connections
4. **Decision rules**: Branch specifications for flowcharts
5. **Retry constraints**: Loop-back paths
6. **Validation rules**: Required steps, forbidden transitions
7. **User intent**: Extracted keywords and intent flags

This context is built dynamically for each request to ensure accurate generation.

## Key Differences from Deterministic Generation

| Aspect | Deterministic | LLM-Based |
|--------|---------------|-----------|
| **Approach** | Rule-based, dataset-driven | Context-aware, flexible |
| **Output** | Always identical for same input | May vary slightly (controlled by temperature) |
| **Validation** | Strict grounding required | Trusts LLM output (no validation) |
| **Flexibility** | Limited to dataset steps | Can infer relationships |
| **Performance** | ~100ms | ~2-5 seconds (Ollama) |
| **Customization** | Cannot extend beyond dataset | Can fine-tune on custom data |

## Model Recommendations

### For Speed:
- **Mistral 7B** - Fast, good quality, recommended
- **Neural-Chat** - Optimized for dialogue

### For Quality:
- **Llama 2 13B** - Better reasoning
- **Llama 2 70B** - Highest quality (requires more VRAM)

### For Balance:
- **OpenHermes 2.5** - Good speed/quality ratio

## Troubleshooting

### Ollama Connection Error

```
LLMWorkflowGenerationError: Cannot connect to Ollama at http://localhost:11434
```

**Fix**: Ensure Ollama is running:
```bash
ollama serve
```

### Invalid JSON Output

The LLM sometimes outputs invalid JSON. The system will:
1. Attempt to extract JSON from markdown code fences
2. Attempt to find raw JSON objects
3. Raise `LLMWorkflowGenerationError` if parsing fails

**Tip**: Lower the `temperature` setting for more consistent JSON output:
```python
"options": {
    "temperature": 0.1,  # Lower = more deterministic
    "top_p": 0.9,
}
```

### Model Out of Memory

If you get OOM errors:
1. Use a smaller model (7B instead of 13B)
2. Increase system RAM
3. Use quantized models (4-bit, 8-bit)

## Running Tests

```bash
# Ensure Ollama is running first
ollama serve &

# Run LLM-specific tests
pytest tests/ -v -k "llm"
```

## Migration from Deterministic to LLM

The system automatically uses LLM generators. The deterministic generators are kept as fallback (not actively used).

To verify LLM generation is working:

```python
response = await pipeline.generate(request)
assert response.workflow.metadata.get("generator") == "llm_workflow_generator"
```

## Performance Tips

1. **Warm up the model** - First request is slower as model loads into memory
2. **Batch requests** - Multiple requests benefit from model caching
3. **Adjust timeout** - Use longer timeouts for larger models
4. **Control temperature** - Lower values = faster, more consistent output
5. **Use smaller models** - Mistral 7B is often better than larger models for this task

## Next Steps

1. Start Ollama: `ollama serve`
2. Run the server: `uvicorn src.api.server:app --reload`
3. Access playground at: http://localhost:5173
4. Generate workflows via the UI or API
5. (Optional) Fine-tune the model with your custom data
