# FrameMind

**Production-grade Video Intelligence Engine** with CLIP-based frame selection and VLM integration.

FrameMind ingests videos, performs intelligent frame selection using computer vision and machine learning, and answers semantic queries using Vision-Language Models (GPT-4V, Claude).

## Features

- **Intelligent Frame Selection**: CLIP embeddings + shot detection reduce thousands of frames to 10-20 key frames
- **Async Processing Pipeline**: Upload → Preprocess → Extract → Analyze → Complete
- **VLM Integration**: Query videos using natural language with GPT-4V or Claude
- **Local-First Architecture**: Run entirely on your machine with Docker
- **Production Ready**: Rate limiting, caching, retries, and structured logging

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   FastAPI   │────▶│    Redis    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Storage   │     │  ARQ Worker │
                    └─────────────┘     └─────────────┘
                                               │
                           ┌───────────────────┼───────────────────┐
                           ▼                   ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                    │    CLIP     │     │    Shot     │     │     VLM     │
                    │   Scorer    │     │  Detector   │     │   Client    │
                    └─────────────┘     └─────────────┘     └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- FFmpeg (for local development)
- Redis 8.0 (for async pipeline and rate limiting)

### Option 1: Docker (Recommended)

```bash
# Clone and enter directory
git clone https://github.com/abhijithreddydasariframemind.git
cd framemind

# Copy environment file
cp .env.example .env

# Start all services
make docker-up

# View logs
make docker-logs
```

The API will be available at `http://localhost:8000`.

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
make dev

# Start Redis (required)
docker run -d -p 6379:6379 redis:8.0-alpine

# Initialize data directories
make init

# Run API server
make api

# In another terminal, run worker
make worker
```

## API Usage

### Upload a Video

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@video.mp4"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Video uploaded successfully. Processing will begin shortly."
}
```

### Check Job Status

```bash
curl "http://localhost:8000/api/v1/ingest/status/550e8400-e29b-41d4-a716-446655440000"
```

### Query a Processed Video

```bash
curl -X POST "http://localhost:8000/api/v1/query/550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is happening in this video?"}'
```

## Project Structure

```
framemind/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # App factory
│   │   ├── routes/       # API endpoints
│   │   └── middleware.py # Rate limiting
│   ├── ml/               # ML/CV modules
│   │   ├── clip_scorer.py    # CLIP embeddings
│   │   ├── shot_detector.py  # Scene detection
│   │   └── frame_selector.py # Intelligent selection
│   ├── workers/          # Async processing
│   │   ├── pipeline.py   # ARQ tasks
│   │   └── orchestrator.py
│   ├── cache/            # Redis caching
│   ├── storage/          # File storage
│   └── core/             # Shared utilities
├── docker/               # Docker configuration
├── tests/                # Test suite
└── scripts/              # Utility scripts
```

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP model |
| `CLIP_DEVICE` | `cpu` | Device for ML (`cpu`, `cuda`, `mps`) |
| `VLM_PROVIDER` | `openai` | VLM provider |
| `VLM_API_KEY` | - | API key for VLM |
| `TARGET_KEYFRAMES` | `30` | Target frames per video |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per window |

See `.env.example` for all options.

## ML Pipeline

### Shot Detection

Uses color histogram analysis to detect scene boundaries:

```python
from src.ml import ShotDetector

detector = ShotDetector()
boundaries = detector.detect_from_video("video.mp4")
# [SceneBoundary(frame_index=120, confidence=0.85), ...]
```

### CLIP Scoring

Computes semantic embeddings for frames:

```python
from src.ml import CLIPScorer

scorer = CLIPScorer()
await scorer.load_model()

embeddings = scorer.embed_frames(frames)
scores = scorer.score_relevance(embeddings, "a person speaking")
```

### Frame Selection

Combines signals for intelligent selection:

```python
from src.ml import FrameSelector

selector = FrameSelector()
await selector.initialize()

result = await selector.select_from_video(
    "video.mp4",
    query="What are the main events?"
)
# Selects ~15 key frames from potentially 1000+
```

## Development

```bash
# Run tests
make test

# Run with coverage
make test-cov

# Lint code
make lint

# Format code
make format

# Type checking
make typecheck
```

## Roadmap

- [ ] VLM integration (GPT-4V, Claude)
- [ ] PostgreSQL support
- [ ] S3 storage backend
- [ ] WebSocket progress updates
- [ ] Batch video processing
- [ ] Audio transcription integration
- [ ] FAISS for large-scale embeddings
