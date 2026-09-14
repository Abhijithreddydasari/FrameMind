# LongShOTBench pilot

Test a small subset before spending on a full run. Downloads and results stay in ignored `data/benchmarks/longshot/`.

The [official dataset](https://huggingface.co/datasets/MBZUAI/longshot-bench) includes speech, environmental audio and visual questions. FrameMind currently searches video only. This adapter selects **single-turn questions marked visual-only**. It does not report an official LongShOT score or measure timestamp recall: these annotations have no retrieval interval labels.

The pinned revision has 3,401 samples, but only **three** meet that filter. One also has an inconsistent `audio_understanding` task label; selection puts it last and records a warning in the references. Review all selected questions before interpreting answer quality.

## Prepare

From the repository root, using Command Prompt and the project environment:

```bat
python -m benchmarks.longshot.benchmark fetch
python -m benchmarks.longshot.benchmark prepare
```

Fetch uses your existing Hugging Face login and requires accepted dataset terms. It downloads annotations only. The default selects at most one video, three questions and 45 total video minutes. In this revision it selects **sample_9048**, one question about the dogs at the beginning of a 39.5-minute recording (`t23Zi0DBSiI`).

Outputs in `data/benchmarks/longshot/pilot/`:

- `manifest.json`: questions and video IDs; no reference answers.
- `references.json`: reference answers, rubrics and metadata warnings for later review.
- `audit.json`: counts, dataset revision and annotation checksum.
- `estimate.json`: source minutes, selected frame budget and hosted retry allowance.

To select all three eligible questions, use `--max-videos 3 --max-questions 3 --max-minutes 120`. This is still a tiny visual subset, not the full benchmark.

## Run

Obtain the selected source recording using its YouTube ID, then ingest the full video through FrameMind. Keep its original timeline. The API and worker must be running, and ingestion must finish first. Source availability is separate from Hugging Face access; the dataset does not include the video files.

Create `data/benchmarks/longshot/pilot/jobs.json`, mapping each source video ID to its completed FrameMind job UUID:

```json
{"t23Zi0DBSiI": "REPLACE_WITH_COMPLETED_JOB_UUID"}
```

Check the mapping against the recording you ingested; the runner cannot infer the YouTube identity of a local file.

```bat
python -m benchmarks.longshot.benchmark run --jobs data/benchmarks/longshot/pilot/jobs.json --output data/benchmarks/longshot/pilot/retrieval.json
```

The default is retrieval-only. Results include latency, evidence, backend status and errors. Answer scores stay null until reviewed; the zeroes in source rubrics are not evaluation results. Cache use is disabled. Use a new output filename for each run.

## Cost and AutoGaze

No model price is assumed. Optionally pass `--usd-per-attempt 0.05` to `prepare` for a scenario using your own assumed cost per hosted call. This allows three attempts per question, but is **not a billing cap**. Indexing, GPU hosting and grading are separate costs. Actual token/image billing varies.

For answer testing, configure the chosen backend and add `--backend default --allow-paid` to `run`. For the optional GPU path, use `--backend nvila_autogaze --allow-paid`. That flag acknowledges possible compute costs; it does not provision a GPU or enable AutoGaze in the server.

Compare answers manually against the separate rubrics. A hosted-model versus AutoGaze comparison changes both model and backend. To isolate pruning speedup, use the same NVILA model and evidence in the existing [paired evaluation harness](../../docs/autogaze.md). This tiny pilot cannot establish accuracy or justify automatic rollout.

Upstream: [LongShOT code and evaluation](https://github.com/mbzuai-oryx/LongShOT). Annotation revision: `5a16213fadcbbbacd40b6feb6e5b048e5c180ddc`.
