"""Metadata storage using SQLAlchemy (SQLite/PostgreSQL)."""

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class VideoJobModel(Base):
    """Video processing job database model."""

    __tablename__ = "video_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Video metadata
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Processing state
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Results
    keyframe_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scene_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class QueryModel(Base):
    """Query history database model."""

    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    query_text: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sources: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FrameEmbeddingModel(Base):
    """Frame embedding storage model."""

    __tablename__ = "frame_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    frame_index: Mapped[int] = mapped_column(Integer, index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(JSON)
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    frame_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TemporalEmbeddingModel(Base):
    """Temporal (clip) embedding database model."""

    __tablename__ = "temporal_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    clip_index: Mapped[int] = mapped_column(Integer)
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[int] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    embedding_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MetadataStore:
    """Async metadata storage using SQLAlchemy.

    Example:
        store = MetadataStore()
        await store.initialize()

        await store.create_job(job_id, {"filename": "video.mp4"})
        job = await store.get_job(job_id)
    """

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or settings.database_url
        self.engine = create_async_engine(self.database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Additive migration: legacy tables and recordings are preserved.
            from sqlalchemy.dialects.postgresql import insert as postgres_insert
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            insert = sqlite_insert if self.engine.dialect.name == "sqlite" else postgres_insert
            await conn.execute(insert(SchemaVersion).values(version=1).on_conflict_do_nothing())

    async def create_job(self, job_id: UUID, data: dict[str, Any]) -> None:
        """Create a new job record."""
        async with self.async_session() as session:
            job = VideoJobModel(
                id=str(job_id),
                status=data.get("status", "pending"),
                video_path=data.get("video_path"),
                filename=data.get("filename"),
                result_data=data.get("result_data"),
            )
            session.add(job)
            await session.commit()

    async def get_job(self, job_id: UUID) -> dict[str, Any] | None:
        """Get job by ID."""
        async with self.async_session() as session:
            job = await session.get(VideoJobModel, str(job_id))
            if job:
                result = {c.name: getattr(job, c.name) for c in VideoJobModel.__table__.columns}
                for key in ("created_at", "updated_at", "completed_at"):
                    if result[key]:
                        result[key] = result[key].isoformat()
                result["pipeline"] = job.result_data or {"needs_reindex": True}
                return result
            return None

    async def update_job(self, job_id: UUID, updates: dict[str, Any]) -> None:
        """Update job fields."""
        async with self.async_session() as session:
            values = {k: v for k, v in updates.items() if k in VideoJobModel.__table__.columns}
            values["updated_at"] = datetime.utcnow()
            statement = update(VideoJobModel).where(VideoJobModel.id == str(job_id))
            # A running task must never undo cancellation.
            if updates.get("status") != "cancelled":
                statement = statement.where(VideoJobModel.status != "cancelled")
            await session.execute(statement.values(**values))
            await session.commit()

    async def save_query(
        self,
        query_id: UUID,
        job_id: UUID,
        query_text: str,
        result: dict[str, Any],
    ) -> None:
        """Save query result."""
        async with self.async_session() as session:
            query = QueryModel(
                id=str(query_id),
                job_id=str(job_id),
                query_text=query_text,
                answer=result.get("answer"),
                confidence=result.get("confidence"),
                frames_analyzed=result.get("frames_analyzed"),
                processing_time_ms=result.get("processing_time_ms"),
                sources=result.get("sources"),
            )
            session.add(query)
            await session.commit()

    async def save_embeddings(
        self,
        job_id: UUID,
        embeddings: Iterable[dict[str, Any]],
    ) -> None:
        """Persist frame embeddings for a job."""
        async with self.async_session() as session:
            embeddings = list(embeddings)
            for item in embeddings:
                await session.execute(
                    delete(FrameEmbeddingModel).where(
                        FrameEmbeddingModel.job_id == str(job_id),
                        FrameEmbeddingModel.frame_index == item["frame_index"],
                    )
                )
            rows = [
                FrameEmbeddingModel(
                    job_id=str(job_id),
                    frame_index=item["frame_index"],
                    timestamp_ms=item.get("timestamp_ms", 0),
                    embedding=item["embedding"],
                    is_selected=item.get("is_selected", False),
                    frame_path=item.get("frame_path"),
                )
                for item in embeddings
            ]
            session.add_all(rows)
            await session.commit()

    async def get_embeddings(self, job_id: UUID) -> list[dict[str, Any]]:
        """Load embeddings for a job."""
        async with self.async_session() as session:
            result = await session.execute(
                FrameEmbeddingModel.__table__.select().where(
                    FrameEmbeddingModel.job_id == str(job_id)
                )
            )
            rows = result.fetchall()

        return [
            {
                "frame_index": row.frame_index,
                "timestamp_ms": row.timestamp_ms,
                "embedding": row.embedding,
                "is_selected": row.is_selected,
                "frame_path": row.frame_path,
            }
            for row in rows
        ]

    async def update_frame_paths(
        self,
        job_id: UUID,
        frame_paths: dict[int, str],
    ) -> None:
        """Update stored frame paths for selected frames."""
        async with self.async_session() as session:
            for frame_index, path in frame_paths.items():
                await session.execute(
                    FrameEmbeddingModel.__table__.update()
                    .where(
                        (FrameEmbeddingModel.job_id == str(job_id))
                        & (FrameEmbeddingModel.frame_index == frame_index)
                    )
                    .values(frame_path=path, is_selected=True)
                )
            await session.commit()

    async def save_temporal_embeddings(
        self,
        job_id: UUID,
        embeddings: list[dict[str, Any]],
    ) -> None:
        """Save temporal (clip) embeddings for a job.

        Args:
            job_id: Job identifier
            embeddings: List of embedding dicts with clip_index, start/end frames, etc.
        """
        async with self.async_session() as session:
            for emb in embeddings:
                await session.execute(
                    delete(TemporalEmbeddingModel).where(
                        TemporalEmbeddingModel.job_id == str(job_id),
                        TemporalEmbeddingModel.clip_index == emb["clip_index"],
                    )
                )
                temp_emb = TemporalEmbeddingModel(
                    job_id=str(job_id),
                    clip_index=emb["clip_index"],
                    start_frame=emb["start_frame"],
                    end_frame=emb["end_frame"],
                    start_ms=emb.get("start_ms", 0),
                    end_ms=emb.get("end_ms", 0),
                    embedding_path=emb.get("embedding_path"),
                )
                session.add(temp_emb)
            await session.commit()

    async def get_temporal_embeddings(
        self,
        job_id: UUID,
    ) -> list[dict[str, Any]]:
        """Get all temporal embeddings for a job."""
        from sqlalchemy import select

        async with self.async_session() as session:
            result = await session.execute(
                select(TemporalEmbeddingModel)
                .where(TemporalEmbeddingModel.job_id == str(job_id))
                .order_by(TemporalEmbeddingModel.clip_index)
            )
            rows = result.scalars().all()

            return [
                {
                    "clip_index": row.clip_index,
                    "start_frame": row.start_frame,
                    "end_frame": row.end_frame,
                    "start_ms": row.start_ms,
                    "end_ms": row.end_ms,
                    "embedding_path": row.embedding_path,
                }
                for row in rows
            ]

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()

    async def save_chunk(self, job_id: str, generation: str, start_ms: int, path: str) -> None:
        async with self.async_session() as session:
            await session.merge(
                ChunkArtifact(job_id=job_id, generation=generation, start_ms=start_ms, path=path)
            )
            await session.commit()

    async def chunks(self, job_id: str, generation: str) -> list[str]:
        async with self.async_session() as session:
            result = await session.execute(
                select(ChunkArtifact.path)
                .where(ChunkArtifact.job_id == job_id, ChunkArtifact.generation == generation)
                .order_by(ChunkArtifact.start_ms)
            )
            return list(result.scalars())

    async def put_query_task(self, query_id: str, job_id: str, request: dict) -> None:
        async with self.async_session() as session:
            session.add(QueryTask(id=query_id, job_id=job_id, request=request, status="pending"))
            await session.commit()

    async def update_query_task(self, query_id: str, **values: Any) -> None:
        async with self.async_session() as session:
            await session.execute(
                update(QueryTask).where(QueryTask.id == query_id).values(**values)
            )
            await session.commit()

    async def get_query_task(self, query_id: str) -> dict | None:
        async with self.async_session() as session:
            task = await session.get(QueryTask, query_id)
            return (
                {c.name: getattr(task, c.name) for c in QueryTask.__table__.columns}
                if task
                else None
            )

    async def unfinished_jobs(self) -> list[dict]:
        async with self.async_session() as session:
            result = await session.execute(
                select(VideoJobModel.id).where(
                    VideoJobModel.status.in_(["pending", "processing", "extracting", "analyzing"])
                )
            )
            ids = list(result.scalars())
        return [job for job_id in ids if (job := await self.get_job(UUID(job_id)))]

    async def acquire_lease(self, job_id: str, token: str, seconds: int) -> bool:
        from sqlalchemy.dialects.postgresql import insert as postgres_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        insert = sqlite_insert if self.engine.dialect.name == "sqlite" else postgres_insert
        now = datetime.utcnow()
        async with self.async_session() as session:
            statement = insert(WorkLease).values(
                job_id=job_id, token=token, expires_at=now + timedelta(seconds=seconds)
            )
            statement = statement.on_conflict_do_update(
                index_elements=["job_id"],
                set_={"token": token, "expires_at": now + timedelta(seconds=seconds)},
                where=WorkLease.expires_at < now,
            )
            result = await session.execute(statement)
            await session.commit()
            return result.rowcount == 1

    async def release_lease(self, job_id: str, token: str):
        async with self.async_session() as session:
            await session.execute(
                delete(WorkLease).where(WorkLease.job_id == job_id, WorkLease.token == token)
            )
            await session.commit()

    async def unfinished_queries(self):
        async with self.async_session() as session:
            result = await session.execute(
                select(QueryTask.id).where(QueryTask.status.in_(["pending", "processing"]))
            )
            return list(result.scalars())


class SchemaVersion(Base):
    __tablename__ = "schema_versions"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)


class ChunkArtifact(Base):
    __tablename__ = "chunk_artifacts"
    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generation: Mapped[str] = mapped_column(String(36), primary_key=True)
    start_ms: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(Text)


class QueryTask(Base):
    __tablename__ = "query_tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    request: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20))
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WorkLease(Base):
    __tablename__ = "work_leases"
    job_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    token: Mapped[str] = mapped_column(String(36))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
