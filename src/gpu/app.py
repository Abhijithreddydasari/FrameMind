"""Run with uvicorn src.gpu.app:app on a private Linux GPU host behind HTTPS."""

import asyncio
import hmac
import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Annotated

import av
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

from src.core.concurrency import run_blocking

app = FastAPI(title="FrameMind experimental video backend")
inference_lock = asyncio.Lock()
runtime = None
MAX_BYTES = 512 * 1024**2


def authorize(authorization: str = Header(default="")):
    token = os.environ.get("AUTOGAZE_TOKEN", "")
    if not token or not hmac.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(401, "Invalid service token")


def read_bundle(path, directory):
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in bundle.infolist()) > MAX_BYTES:
            raise ValueError("Invalid or oversized evidence bundle")
        if bundle.getinfo("metadata.json").file_size > 1024 * 1024:
            raise ValueError("Oversized metadata")
        metadata = json.loads(bundle.read("metadata.json"))
        timestamps = metadata["timestamps_ms"]
        if (
            not isinstance(timestamps, list)
            or not 1 <= len(timestamps) <= 1024
            or any(type(t) is not int or t < 0 for t in timestamps)
            or timestamps != sorted(set(timestamps))
        ):
            raise ValueError("Invalid source timestamps")
        expected = {"metadata.json", *(f"frame-{i:04d}.jpg" for i in range(len(timestamps)))}
        if set(names) != expected:
            raise ValueError("Unexpected archive entries")
        if (
            type(metadata.get("gaze")) is not bool
            or type(metadata.get("max_tiles")) is not int
            or not 1 <= metadata["max_tiles"] <= 48
        ):
            raise ValueError("Invalid inference settings")
        if not isinstance(metadata.get("prompt"), str) or len(metadata["prompt"]) > 100000:
            raise ValueError("Invalid prompt")
        paths = []
        for i in range(len(timestamps)):
            name = f"frame-{i:04d}.jpg"
            image_path = directory / name
            image_path.write_bytes(bundle.read(name))
            with Image.open(image_path) as image:
                if image.width * image.height > 4096 * 4096:
                    raise ValueError("Frame exceeds supported resolution")
                image.verify()
            paths.append(image_path)
        return metadata, paths


def make_video(paths, destination):
    # The processor requires 16-frame groups. Repeat the final sample, never invent timestamps.
    count = ((len(paths) + 15) // 16) * 16
    with Image.open(paths[0]) as image:
        width, height = image.size
    with av.open(str(destination), "w") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for i in range(count):
            with Image.open(paths[min(i, len(paths) - 1)]) as image:
                if image.size != (width, height):
                    raise ValueError("Evidence frames have inconsistent dimensions")
                frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB")), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return count


class NVILARuntime:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")
        self.torch = torch
        self.revision = os.environ.get("NVILA_REVISION", "")
        self.gaze_revision = os.environ.get("AUTOGAZE_REVISION", "")
        if not all(
            re.fullmatch(r"[0-9a-f]{40}", revision)
            for revision in (self.revision, self.gaze_revision)
        ):
            raise RuntimeError("Pin NVILA_REVISION and AUTOGAZE_REVISION to reviewed commit hashes")
        model_id = "nvidia/NVILA-8B-HD-Video"
        # Load reviewed AutoGaze weights locally so processor nested loads cannot follow main.
        from huggingface_hub import snapshot_download

        gaze_path = snapshot_download("nvidia/AutoGaze", revision=self.gaze_revision)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=self.revision,
            trust_remote_code=True,
            autogaze_model_id=gaze_path,
            max_batch_size_autogaze=1,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            max_batch_size_siglip=1,
        ).eval()

    def infer(self, paths, metadata, directory):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        video_path = directory / "samples.mp4"
        count = make_video(paths, video_path)
        self.processor.num_video_frames = count
        self.processor.num_video_frames_thumbnail = count
        self.processor.max_tiles_video = metadata["max_tiles"]
        self.processor.gazing_ratio_tile = 0.75 if metadata["gaze"] else 1
        self.processor.task_loss_requirement_tile = 0.6 if metadata["gaze"] else None
        self.processor.gazing_ratio_thumbnail = 1
        self.processor.task_loss_requirement_thumbnail = None
        padded_timestamps = metadata["timestamps_ms"] + [metadata["timestamps_ms"][-1]] * (
            count - len(paths)
        )
        prompt = (
            metadata["prompt"]
            + "\nVideo frame source timestamps, in order: "
            + json.dumps(padded_timestamps)
        )
        with torch.inference_mode():
            inputs = self.processor(
                text=f"{self.processor.tokenizer.video_token}\n{prompt}",
                videos=str(video_path),
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            torch.cuda.synchronize()
            preprocess_ms = round((time.perf_counter() - started) * 1000)
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
            torch.cuda.synchronize()
            content = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
        return {
            "content": content,
            "timestamps_ms": metadata["timestamps_ms"],
            "metrics": {
                "gaze": metadata["gaze"],
                "sampled_frames": len(paths),
                "padded_frames": count - len(paths),
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
                "preprocess_ms": preprocess_ms,
                "inference_total_ms": round((time.perf_counter() - started) * 1000),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "model_revision": self.revision,
                "autogaze_revision": self.gaze_revision,
            },
        }


@app.get("/health", dependencies=[Depends(authorize)])
async def health():
    import torch

    return {
        "cuda": torch.cuda.is_available(),
        "loaded": runtime is not None,
        "free_vram_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0,
        "max_frames": 1024,
        "max_tiles": 48,
        "supports_no_pruning": True,
        "model_revision": os.environ.get("NVILA_REVISION"),
        "research_only": True,
    }


@app.post("/preflight", dependencies=[Depends(authorize)])
async def preflight():
    global runtime
    async with inference_lock:
        if runtime is None:
            runtime = await run_blocking(NVILARuntime)
    return await health()


@app.post("/analyze", dependencies=[Depends(authorize)])
async def analyze(file: Annotated[UploadFile, File(...)]):
    global runtime
    async with inference_lock:
        with tempfile.TemporaryDirectory(prefix="framemind-gpu-") as temp:
            directory = Path(temp)
            archive = directory / "evidence.zip"
            try:
                size = 0
                with archive.open("wb") as handle:
                    while block := await file.read(1024**2):
                        size += len(block)
                        if size > MAX_BYTES:
                            raise HTTPException(413, "Evidence exceeds transfer limit")
                        handle.write(block)
                metadata, paths = await run_blocking(read_bundle, archive, directory)
                if runtime is None:
                    runtime = await run_blocking(NVILARuntime)
                return await run_blocking(runtime.infer, paths, metadata, directory)
            except (ValueError, KeyError, zipfile.BadZipFile) as exc:
                raise HTTPException(422, str(exc)) from exc
            except RuntimeError as exc:
                # No automatic downsampling or model/provider fallback.
                raise HTTPException(503, str(exc)) from exc
            finally:
                await file.close()
