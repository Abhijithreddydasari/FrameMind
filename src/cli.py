"""Register an existing local recording without copying it."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from arq.connections import create_pool

from src.core.config import settings
from src.ingest.stream import fingerprint, probe
from src.storage.metadata import MetadataStore
from src.workers.pipeline import get_redis_settings


async def ingest(path):
    path = Path(path).resolve(strict=True)
    probe(path)
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    store = MetadataStore()
    await store.initialize()
    queue = None
    job_id = uuid4()
    try:
        await store.create_job(
            job_id,
            {
                "filename": path.name,
                "video_path": str(path),
                "result_data": {
                    "needs_reindex": False,
                    "source": fingerprint(path),
                    "externally_owned": True,
                },
            },
        )
        queue = await create_pool(get_redis_settings())
        await queue.enqueue_job(
            "process_video", str(job_id), str(path), _job_id=f"ingest:{job_id}:new:0"
        )
        return {"job_id": str(job_id), "status": "pending", "source": str(path)}
    except Exception:
        await store.update_job(
            job_id, {"status": "failed", "error": "Unable to enqueue local recording"}
        )
        raise
    finally:
        if queue:
            await queue.aclose()
        await store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(ingest(args.path))))


if __name__ == "__main__":
    main()
