# Optional AutoGaze backend

FrameMind retrieves intervals, then sends selected evidence to NVILA. AutoGaze skips redundant image patches during analysis. It does not track objects or detect incidents. Speed gains depend on repeated content, transfer time and how much useful detail survives pruning.

## Separate Linux GPU host

Use a Linux NVIDIA host with Docker's GPU runtime and enough VRAM for the selected profile. The container uses CUDA 12.8, PyTorch 2.7.1, Transformers 4.57.6, and FlashAttention 2.7.4.post1. Python dependencies are resolved in `docker/requirements-gpu.lock`; the native attention extension still needs a host build and smoke test. Start with one inference at a time, 128 samples per interval, and 12 tiles. No minimum-VRAM performance claim has been measured for this profile.

Obtain reviewed, immutable commit hashes for the [AutoGaze code](https://github.com/NVlabs/AutoGaze), [NVILA checkpoint](https://huggingface.co/nvidia/NVILA-8B-HD-Video), and [AutoGaze checkpoint](https://huggingface.co/nvidia/AutoGaze). Set them on the GPU host:

```sh
export AUTOGAZE_COMMIT=REVIEWED_CODE_COMMIT
export NVILA_REVISION=REVIEWED_MODEL_COMMIT
export AUTOGAZE_REVISION=REVIEWED_GAZE_COMMIT
export AUTOGAZE_TOKEN=YOUR_RANDOM_SERVICE_TOKEN
docker compose -f docker/docker-compose.gpu.yml up --build -d
curl -X POST -H "Authorization: Bearer $AUTOGAZE_TOKEN" http://127.0.0.1:8080/preflight
```

The checkpoint loader executes pinned upstream model code (`trust_remote_code`). Review those revisions before using them. Preflight downloads/loads the models and reports CUDA, free VRAM, and loaded status. It does not replace a real inference test. Use a TLS reverse proxy or an SSH tunnel; port 8080 binds to loopback and the health, preflight, and analysis endpoints require a bearer token. The client rejects plain HTTP except localhost/127.0.0.1.

Set local `.env` values and restart the API/worker:

```ini
AUTOGAZE_ENABLED=true
AUTOGAZE_URL=https://YOUR_PRIVATE_GPU_ENDPOINT
AUTOGAZE_TOKEN=THE_SAME_SERVICE_TOKEN
VIDEO_MAX_FRAMES=128
VIDEO_MAX_TILES=12
```

For an SSH tunnel, use `http://127.0.0.1:8080` when running the API directly. Container loopback addresses the container itself, so use a reachable HTTPS endpoint for Compose.

Request `analysis_backend: "nvila_autogaze"` on a query or inspection. Enabling the setting does not alter the default backend. OOM, transfer errors, and invalid answers become explicit analysis failures; there is no automatic downsampling, provider switch, or paid fallback.

Evidence bundles contain selected JPEGs, actual source timestamps, and the question. Each interval is reconstructed as a lossless video for the upstream processor, with repeated final samples to satisfy 16-frame grouping. The prompt carries the original timestamp mapping. Padding is reported separately. Source files never leave the local host in this path. Metrics include sampled/padded frames, token counts, preprocessing/inference time, transfer size, remote elapsed time, peak allocated VRAM, and model revisions. Token counts describe model input/output, not a separately measured ViT FLOP count.

## Paired evaluation

Create an evaluation manifest like `evaluation/manifest.example.json`, using actual local video paths and completed job IDs. Use independent recordings for tuning and held-out evaluation. Include short events, small objects, occlusion, poor lighting, camera motion, repeated backgrounds, and questions where the correct response is insufficient evidence. Mark safety-relevant events as critical. Do not tune on the held-out set.

Warm the models on tuning cases first. Keep model revisions, frame and tile budgets, and footage fixed for each pair. Repeat runs in alternating order to check warm-up and load effects. The harness disables answer caching and records errors.

```sh
python -m scripts.evaluate run evaluation/manifest.json --mode retrieve --output evaluation/retrieval.json
python -m scripts.evaluate run evaluation/manifest.json --mode default --output evaluation/default.json
python -m scripts.evaluate run evaluation/manifest.json --mode pipeline-gaze --output evaluation/pipeline-gaze.json
python -m scripts.evaluate run evaluation/manifest.json --mode no-gaze --output evaluation/no-gaze.json
python -m scripts.evaluate run evaluation/manifest.json --mode gaze --output evaluation/gaze.json
```

`gaze` and `no-gaze` use the same annotated intervals and sampling, bypassing retrieval to isolate inference. `default` and `pipeline-gaze` include retrieval, transfer, and analysis for the real pipeline comparison. The hosted default and NVILA are different models, so assess both answer quality and latency separately from the pruning ablation. `retrieve` records interval-hit recall at up to ten returned sources; the normal configuration returns at most five merged intervals. This metric is not object-detection recall.

Review every answer against annotated footage. Create two score files, one per comparison. Each maps case IDs to four boolean values, as in `evaluation/scores.example.json`; missed means the queried annotated event was missed. Use both false for correct negative examples. Review sources as well as answer wording. The evaluator does not use an LLM judge.

```sh
python -m scripts.evaluate report --baseline evaluation/no-gaze.json --accelerated evaluation/gaze.json --scores evaluation/ablation-scores.json --pipeline-baseline evaluation/default.json --pipeline-accelerated evaluation/pipeline-gaze.json --pipeline-scores evaluation/pipeline-scores.json --output evaluation/report.json
```

Eligibility requires both comparisons to pass: at least 100 paired held-out questions from at least three recordings, all manually scored, no failed runs, no additional critical misses, no more than two percentage points of accuracy loss, and at least 1.25x p95 latency improvement. The pruning ablation also requires matching timestamps, frame/tile budgets, and model revisions. Without the pipeline measurements, eligibility stays false. These are engineering acceptance thresholds, not statistical guarantees. Examine per-category results and repeated runs before deciding.

Enable AutoGaze manually only after reviewing both comparisons; the evaluator never changes settings. Faster inference alone does not prove faster search. Commercial use also requires resolving the checkpoint's noncommercial/research terms; see the [model card](https://huggingface.co/nvidia/NVILA-8B-HD-Video).

For a small public-data trial, use the [LongShOTBench pilot](../benchmarks/longshot/README.md). Its visual-only subset is too small for these promotion gates.
