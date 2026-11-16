from __future__ import annotations

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from supportflow.config import Settings


def create_pg_pool(settings: Settings) -> AsyncConnectionPool[Any]:
    """创建符合 LangGraph AsyncPostgresSaver 与事务需求的异步连接池。

    注意：autocommit=True 与 prepare_threshold=0 是 AsyncPostgresSaver.setup() 正常运行的必要参数；
    业务侧仍可通过 `async with conn.transaction():` 显式开启事务块（BEGIN/COMMIT/ROLLBACK）。
    """
    return AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c statement_timeout={settings.db_statement_timeout_ms}",
        },
    )


async def init_database_schema(
    pool: AsyncConnectionPool[Any],
    *,
    embedding_dim: int = 1024,
) -> None:
    """初始化 pgvector 扩展、知识分块表与工单幂等表。"""
    async with pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                document_id TEXT NOT NULL,
                document_version TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({embedding_dim}) NOT NULL,
                PRIMARY KEY (document_id, document_version, chunk_id)
            );
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_version
            ON knowledge_chunks (document_version);
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT tickets_idempotency_unique UNIQUE (account_id, idempotency_key)
            );
            """
        )


async def create_postgres_checkpointer(
    pool: AsyncConnectionPool[Any],
) -> AsyncPostgresSaver:
    """初始化并返回 LangGraph PostgreSQL 持久化 Checkpointer。"""
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return checkpointer
