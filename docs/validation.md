# Validation record

Measured locally on Windows, Python 3.11.9. This file separates deterministic pipeline checks from model and deployment validation.

## Long-recording resource acceptance

`FM_RUN_RESOURCE_TEST=1 pytest tests/integration/test_resources.py -q -s` passed on 2026-09-14.

- Source: two hours, 1920 x 1080, 2 FPS; generated H.264 video.
- Complete indexed coverage: 7,200,000 ms.
- Event in the final ten seconds: retrieved in the top ten spatial hits.
- Peak sampled process RSS during ingestion: 390,873,088 bytes (372.8 MiB).
- Ingestion time: 163.19 seconds; total test time including fixture generation: 406.80 seconds.
- Spatial and temporal encoders: deterministic test doubles. These numbers exclude real CLIP/X-CLIP inference costs and do not estimate 30 FPS traffic-video throughput.

The resource report is written to ignored `data/resource-report.json`. The default suite skips this test; rerun it explicitly after changing decoding or batching.

## Regression coverage

The locked CPU environment passed 66 tests, with the two opt-in tests excluded from the default run. Ruff lint and formatting checks passed, as did Python compilation and dependency compatibility checks. The initial real-model attempt revealed a conflicting system TorchVision; the project environment was then isolated from system packages. The standalone mypy launcher was blocked by Windows Application Control, so strict typing is not included in the passing checks.

The suite exercises actual PyAV decoding, nonzero stream origins, source timestamps, late-event retrieval, persisted index reuse, bounded batches, chunk retries, cancellation/source preservation, temporal fallback warnings, legacy reindex requirements, precise inspection, citation validation, backend failure visibility, queue failure persistence, asynchronous query completion, SQL leases, GPU bundle validation/authentication, and benchmark promotion gates. Model calls and queue calls are replaced where a test is intended to isolate orchestration.

## Real model smoke test

`scripts\test-models.cmd --offline` passed separately (11.74 seconds) after downloading the checkpoints into the project cache. It loads both models on CPU, checks compatible image/video and text embedding dimensions, encodes a short real video through the chunk encoder, saves both streams, and retrieves from the persisted FAISS indexes.

- CLIP: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- X-CLIP: `microsoft/xclip-base-patch32`, revision `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`.
- Runtime: CPU PyTorch 2.14.0 and Transformers 4.57.6 from `requirements.lock`.

This exposed and verified fixes for X-CLIP video preprocessing and Transformers' tuple-return mismatch in temporal integration. The test verifies execution and index compatibility; it does not establish surveillance accuracy or answer quality. No LLM is called.

## Deployment checks still required

- Build/run the CPU Compose stack with a running Docker engine and verify Redis/ARQ across separate API and worker processes.
- Build the Linux GPU image, run preflight, and complete a real NVILA inference in both gaze modes on the target GPU.
- Run the held-out, manually reviewed evaluation described in `autogaze.md` before promoting AutoGaze.
- Test on representative user recordings for retrieval accuracy, brief/small-object misses, query latency, inference memory, and decoder throughput.

The local machine's Docker CLI is available, but its Docker engine was not running during implementation. No cloud host was provisioned, no hosted VLM call was made, and no GPU speedup or answer-quality result is claimed.
