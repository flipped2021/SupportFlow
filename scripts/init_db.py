from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 确保直接执行 `python scripts/init_db.py` 时也能正确解析 src/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from supportflow.config import get_settings  # noqa: E402
from supportflow.env_factory import create_embedder  # noqa: E402
from supportflow.persistence.db import (  # noqa: E402
    create_pg_pool,
    create_postgres_checkpointer,
    init_database_schema,
)
from supportflow.tools.knowledge import DEFAULT_SAAS_DOCUMENTS, _vector_to_pg_literal  # noqa: E402


async def run_init(*, use_test_embedder: bool = False) -> None:
    settings = get_settings()
    embed_dim = 128 if use_test_embedder else settings.embedding_dim
    embedder = create_embedder(settings, use_test_embedder=use_test_embedder)

    pool = create_pg_pool(settings)
    await pool.open(wait=True)
    try:
        print(f"1/3 初始化数据库表结构 (embedding_dim={embed_dim})...")
        await init_database_schema(pool, embedding_dim=embed_dim)

        print("2/3 初始化 LangGraph PostgreSQL Checkpointer...")
        await create_postgres_checkpointer(pool)

        print(f"3/3 向量化并写入 {len(DEFAULT_SAAS_DOCUMENTS)} 条示例 SaaS 产品文档分块...")
        texts = [f"{doc.title}\n{doc.text}" for doc in DEFAULT_SAAS_DOCUMENTS]
        vectors = await embedder.embed_documents(texts)

        async with pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    for doc, vec in zip(DEFAULT_SAAS_DOCUMENTS, vectors, strict=True):
                        await cur.execute(
                            """
                            INSERT INTO knowledge_chunks (
                                document_id, document_version, chunk_id, title, content, embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s::vector)
                            ON CONFLICT (document_id, document_version, chunk_id)
                            DO UPDATE SET
                                title = EXCLUDED.title,
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding;
                            """,
                            (
                                doc.document_id,
                                doc.document_version,
                                doc.chunk_id,
                                doc.title,
                                doc.text,
                                _vector_to_pg_literal(vec),
                            ),
                        )
        print("数据库初始化完成。")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 SupportFlow PostgreSQL 数据库与示例知识库")
    parser.add_argument(
        "--test-embedder",
        action="store_true",
        help="使用 128 维确定性本地向量器（跳过下载 BGE-M3 模型，适合快速联调）",
    )
    args = parser.parse_args()
    asyncio.run(run_init(use_test_embedder=args.test_embedder))


if __name__ == "__main__":
    main()
