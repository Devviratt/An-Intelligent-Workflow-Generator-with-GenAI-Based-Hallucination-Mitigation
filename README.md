# Intelligent Workflow Generator with GenAI-Based Hallucination Mitigation

A production-level, dataset-driven workflow generation engine that synthesizes
structured workflows from domain knowledge — with **zero external AI API dependencies**.

## Architecture

```
User Input → Instruction Parser → Domain Dataset Engine → Workflow Generator
                                                              ↓
                        Rendered Output ← Layout Engine ← Validation Engine ← Hallucination Mitigation
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `instruction_parser` | NLP-based intent extraction via TF-IDF + keyword matching |
| `domain_engine` | Structured JSON dataset management per domain |
| `workflow_generator` | Deterministic workflow synthesis from datasets |
| `hallucination_mitigation` | Grounding-based node/transition validation |
| `validation_engine` | Schema, logic, dependency, cycle, and depth validation |
| `layout_engine` | BFS-based deterministic coordinate assignment |
| `local_model` | Optional Ollama-based RAG with strict output gating |
| `pipeline` | Async orchestrator connecting all stages |
| `api` | FastAPI REST interface with versioned routing |
| `observability` | Stage profiling, hallucination metrics, explainability, evaluation |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run server
uvicorn src.api.server:app --reload --port 8000

# Run tests
pytest tests/ -v
```

## API Usage

All endpoints are under the `/api/v1/` prefix. Interactive docs at `/docs`.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page (HTML) or service info (JSON) |
| `GET` | `/api/v1/health` | Health check with dataset count and supported modes |
| `POST` | `/api/v1/generate` | Generate a workflow or flowchart from instruction |
| `POST` | `/api/v1/validate` | Validate an existing workflow |
| `POST` | `/api/v1/evaluate` | Generate with full observability / evaluation data |
| `GET` | `/api/v1/domains` | List all available domain datasets |
| `GET` | `/api/v1/domains/{domain}` | Get full detail for a single domain |

### Examples

```bash
# Service info (JSON)
curl http://localhost:8000/

# Health check
curl http://localhost:8000/api/v1/health

# Generate a workflow
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Create an online payment processing workflow"}'

# Generate a flowchart
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "payment processing", "mode": "flowchart", "domain_hint": "online_payment"}'

# Generate with evaluation mode (full observability)
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Create an online payment processing workflow"}'

# List available domains
curl http://localhost:8000/api/v1/domains

# Validate an existing workflow
curl -X POST http://localhost:8000/api/v1/validate \
  -H "Content-Type: application/json" \
  -d @workflow.json
```

## Design Principles

- **No external AI APIs** — all intelligence is dataset-driven
- **Deterministic output** — same input always produces same workflow
- **Grounded generation** — every node/edge validated against domain dataset
- **Single-pass layout** — BFS depth + branch index, no recursion
- **Async-first** — no synchronous blocking in the pipeline
- **Strict separation** — generation, validation, layout, rendering are independent

## API Architecture

```
src/api/
├── server.py           # App factory, lifespan, CORS, middleware
├── exceptions.py       # Global structured exception handlers
├── middleware.py        # Request logging middleware
├── landing.py          # Root endpoint — glassmorphism HTML + JSON
└── routers/
    ├── workflows.py    # POST /generate, /validate, /evaluate
    ├── domains.py      # GET /domains, /domains/{domain}
    └── health.py       # GET /health
```

### Production Features

- **Versioned routes** — all endpoints under `/api/v1/`
- **Structured error responses** — consistent JSON for all error types
- **Method-not-allowed handling** — clear guidance on correct HTTP method
- **Request logging** — method, path, duration_ms, status_code per request
- **CORS** — locked to `localhost:3000` and `localhost:5173`
- **Swagger metadata** — title, description, contact, license, terms of service
- **Glassmorphism landing page** — professional developer experience at `/`
