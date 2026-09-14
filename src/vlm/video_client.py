"""Explicit remote video inference; only selected evidence leaves the local host."""

import json
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx


class RemoteVideoClient:
    def __init__(self, config):
        self.config = config
        parsed = urlparse(config.autogaze_url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1")
        ):
            raise ValueError("GPU endpoint must use HTTPS (HTTP allowed only on loopback)")
        if not config.autogaze_token:
            raise ValueError("GPU service token is required")

    async def analyze(self, paths, source, prompt, *, gaze=True):
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="framemind-transfer-") as directory:
            archive = Path(directory) / "evidence.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                for i, path in enumerate(paths):
                    bundle.write(path, f"frame-{i:04d}.jpg")
                bundle.writestr(
                    "metadata.json",
                    json.dumps(
                        {
                            "prompt": prompt,
                            "gaze": gaze,
                            "timestamps_ms": source.frame_timestamps_ms,
                            "max_tiles": self.config.video_max_tiles,
                            "evidence_id": source.evidence_id,
                        }
                    ),
                )
            async with httpx.AsyncClient(timeout=self.config.autogaze_timeout) as client:
                with archive.open("rb") as handle:
                    response = await client.post(
                        self.config.autogaze_url.rstrip("/") + "/analyze",
                        headers={"Authorization": f"Bearer {self.config.autogaze_token}"},
                        files={"file": ("evidence.zip", handle, "application/zip")},
                    )
                response.raise_for_status()
                result = response.json()
            if result.get("timestamps_ms") != source.frame_timestamps_ms:
                raise ValueError("GPU backend returned mismatched evidence timestamps")
            result["metrics"]["remote_total_ms"] = round((time.perf_counter() - started) * 1000)
            result["metrics"]["transfer_bytes"] = archive.stat().st_size
            return result
