from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import httpx
from psycopg_pool import AsyncConnectionPool
from rank_bm25 import BM25Okapi

from supportflow.deps import KnowledgeRetrieverProtocol
from supportflow.schemas import ReadDocumentChunkArgs, SearchDocumentsArgs
from supportflow.tools import FaultInjector

# 匹配完整技术标识（例如 storage:write, E_UPLOAD_403, v2.3, req-123, STRICT_SCOPE_CHECK_V23）
_TECH_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:\.\-/]*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_preserving_identifiers(text: str) -> list[str]:
    """保留技术标识的混合中英文分词器：完整技术标识符 + 子词拆分 + 中文二元切片。"""
    tokens: list[str] = []

    for match in _TECH_TOKEN_RE.finditer(text):
        raw = match.group(0).lower()
        tokens.append(raw)
        parts = [p for p in re.split(r"[:_\.\-/]+", raw) if p]
        if len(parts) > 1:
            tokens.extend(parts)

    for match in _CJK_RE.finditer(text):
        seg = match.group(0)
        for ch in seg:
            tokens.append(ch)
        for i in range(len(seg) - 1):
            tokens.append(seg[i : i + 2])

    return tokens or [text.lower()]


@dataclass(frozen=True, slots=True)
class KnowledgeChunkRecord:
    document_id: str
    document_version: str
    chunk_id: str
    title: str
    text: str


DEFAULT_SAAS_DOCUMENTS: list[KnowledgeChunkRecord] = [
    KnowledgeChunkRecord(
        document_id="upload-auth-guide",
        document_version="v2.3",
        chunk_id="chunk-01",
        title="v2.3 文件上传接口权限规范",
        text=(
            "在云存储 SaaS v2.3 版本中，POST /api/v2/files/upload 与分片上传接口"
            "强制启用 STRICT_SCOPE_CHECK_V23 校验。调用方令牌必须同时具备 "
            "storage:read 与 storage:write 权限范围（token_scopes）。"
            "若账户或令牌仅包含 storage:read 缺少 storage:write，网关将直接返回 "
            "HTTP 403，错误码为 E_UPLOAD_403。"
        ),
    ),
    KnowledgeChunkRecord(
        document_id="upload-auth-guide",
        document_version="v2.3",
        chunk_id="chunk-02",
        title="v2.3 升级变更与兼容提示",
        text=(
            "从 v2.2 升级至 v2.3 后，旧版隐式写权限兼容开关 IMPLICIT_WRITE_LEGACY 已下线。"
            "部分历史 API Token 在 v2.2 中虽仅配置了 storage:read 仍可上传，"
            "但在 v2.3 必须重新签发包含 storage:write 的令牌并在授权后复测上传。"
        ),
    ),
    KnowledgeChunkRecord(
        document_id="upload-auth-guide",
        document_version="v2.2",
        chunk_id="chunk-01",
        title="v2.2 文件上传接口权限说明",
        text=(
            "在 v2.2 版本中，默认开启 IMPLICIT_WRITE_LEGACY 兼容模式，"
            "普通活跃账户的默认读写令牌可访问上传接口。若出现 403 且错误码为 "
            "E_ACCOUNT_SUSPENDED，请先核实账户是否处于 suspended 状态。"
        ),
    ),
    KnowledgeChunkRecord(
        document_id="quota-and-limits",
        document_version="v2.3",
        chunk_id="chunk-01",
        title="存储空间配额与 API 频率限制",
        text=(
            "当账户已用存储空间达到 storage_limit_bytes 上限，或当日请求量超过 "
            "daily_api_limit 时，上传请求将被拒绝并返回 HTTP 429 或 HTTP 403，"
            "错误码分别为 E_QUOTA_STORAGE_FULL 与 E_RATE_LIMIT_EXCEEDED。"
            "排查时需通过配额查询工具核实 quota_exceeded 字段，并注意单分片上限 MAX_PART_SIZE_MB=64。"
        ),
    ),
    KnowledgeChunkRecord(
        document_id="error-code-reference",
        document_version="v2.3",
        chunk_id="chunk-01",
        title="常见 403/5xx 上传错误码对照表",
        text=(
            "1) E_UPLOAD_403：令牌缺少 storage:write 或账户权限受限；\n"
            "2) E_ACCOUNT_RESTRICTED：账户欠费或风控冻结，status 为 suspended/restricted；\n"
            "3) E_GATEWAY_REGION_DEGRADED：区域上传网关（upload/gateway）降级，通常返回 503 并关联 incident_id；\n"
            "4) E_AUTH_TOKEN_EXPIRED：签名过期返回 401，需刷新凭证。"
        ),
    ),
    KnowledgeChunkRecord(
        document_id="multipart-upload-handbook",
        document_version="v2.3",
        chunk_id="chunk-01",
        title="大文件分片上传与超时重试配置",
        text=(
            "超过 64MB 的文件需使用 /api/v2/files/multipart/init 初始化分片，"
            "配置项 MULTIPART_UPLOAD_TIMEOUT_MS 默认为 30000ms。"
            "若服务端区域网关健康状态为 degraded，建议切换备用区域或等待事件恢复后再复测。"
        ),
    ),
]


class EmbedderProtocol(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的本地 BGE-M3 真实向量编码器（异步非阻塞加载与推理）。"""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = await asyncio.to_thread(SentenceTransformer, self._model_name)
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._ensure_model()
        vectors = await asyncio.to_thread(
            model.encode,
            list(texts),
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in vectors]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class OpenAICompatibleEmbedder:
    """兼容 OpenAI Embeddings 接口的远程 BGE-M3 推理服务客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-m3",
        timeout_s: float = 15.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": list(texts)},
            )
            resp.raise_for_status()
            payload = resp.json()
        items = sorted(payload["data"], key=lambda x: x["index"])
        return [list(item["embedding"]) for item in items]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class DeterministicTestEmbedder:
    """仅供离线单元测试与无 GPU 环境使用的确定性字符特征哈希向量器。"""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = tokenize_preserving_identifiers(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._encode_one(text)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_rrf_fusion(
    vector_ranked: list[tuple[KnowledgeChunkRecord, float]],
    bm25_ranked: list[tuple[KnowledgeChunkRecord, float]],
    *,
    rrf_k: int = 60,
) -> list[tuple[KnowledgeChunkRecord, float]]:
    """倒数排名融合（RRF）：1 / (k + rank)，仅对实际命中的分块计分。"""
    scores: dict[tuple[str, str, str], float] = {}
    records: dict[tuple[str, str, str], KnowledgeChunkRecord] = {}

    for ranking in (vector_ranked, bm25_ranked):
        for rank, (chunk, _) in enumerate(ranking, start=1):
            key = (chunk.document_id, chunk.document_version, chunk.chunk_id)
            records[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ordered_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [(records[k], scores[k]) for k in ordered_keys]


def _chunk_to_dict(chunk: KnowledgeChunkRecord, score: float | None = None) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "document_version": chunk.document_version,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "text": chunk.text,
        "score": round(score, 6) if score is not None else None,
    }


class InMemoryHybridKnowledgeRetriever(KnowledgeRetrieverProtocol):
    """本地混合检索器：向量检索 + BM25 + RRF 融合排序 + 产品版本过滤 + 故障注入。"""

    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        chunks: Sequence[KnowledgeChunkRecord] | None = None,
        rrf_k: int = 60,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self._embedder = embedder
        self._chunks = list(chunks or DEFAULT_SAAS_DOCUMENTS)
        self._rrf_k = rrf_k
        self._fault_injector = fault_injector
        self._chunk_vectors: list[list[float]] | None = None
        self._bm25: BM25Okapi | None = None
        self._index_lock = asyncio.Lock()

    async def _ensure_index(self) -> None:
        if self._chunk_vectors is not None and self._bm25 is not None:
            return
        async with self._index_lock:
            if self._chunk_vectors is None:
                texts = [f"{c.title}\n{c.text}" for c in self._chunks]
                self._chunk_vectors = await self._embedder.embed_documents(texts)
            if self._bm25 is None:
                corpus = [
                    tokenize_preserving_identifiers(f"{c.title}\n{c.text}")
                    for c in self._chunks
                ]
                self._bm25 = BM25Okapi(corpus) if corpus else None

    async def search_documents(self, args: SearchDocumentsArgs) -> dict[str, Any]:
        if self._fault_injector is not None:
            rule = await self._fault_injector.maybe_inject_tool_fault("search_documents")
            if rule is not None and rule.fault_type == "invalid_output_schema":
                return {"chunks": [{"text": "缺少来源字段的畸形返回体"}]}

        await self._ensure_index()
        assert self._chunk_vectors is not None

        candidate_indices = [
            idx
            for idx, chunk in enumerate(self._chunks)
            if chunk.document_version == args.product_version
        ]
        if not candidate_indices:
            return {"chunks": [], "total_candidates": 0}

        query_vec = await self._embedder.embed_query(args.query)
        vector_scored = [
            (
                self._chunks[idx],
                _cosine_similarity(query_vec, self._chunk_vectors[idx]),
            )
            for idx in candidate_indices
        ]
        vector_ranked = sorted(vector_scored, key=lambda item: item[1], reverse=True)

        bm25_ranked: list[tuple[KnowledgeChunkRecord, float]] = []
        if self._bm25 is not None:
            query_tokens = tokenize_preserving_identifiers(args.query)
            all_bm25_scores = self._bm25.get_scores(query_tokens)
            bm25_scored = [
                (self._chunks[idx], float(all_bm25_scores[idx]))
                for idx in candidate_indices
                if float(all_bm25_scores[idx]) > 0.0
            ]
            bm25_ranked = sorted(bm25_scored, key=lambda item: item[1], reverse=True)

        recall_depth = max(args.top_k * 2, 10)
        fused = compute_rrf_fusion(
            vector_ranked[:recall_depth],
            bm25_ranked[:recall_depth],
            rrf_k=self._rrf_k,
        )
        top_items = fused[: args.top_k]
        return {
            "chunks": [_chunk_to_dict(chunk, score) for chunk, score in top_items],
            "total_candidates": len(candidate_indices),
        }

    async def read_document_chunk(self, args: ReadDocumentChunkArgs) -> dict[str, Any]:
        if self._fault_injector is not None:
            rule = await self._fault_injector.maybe_inject_tool_fault("read_document_chunk")
            if rule is not None and rule.fault_type == "invalid_output_schema":
                return {"found": True}

        for chunk in self._chunks:
            if (
                chunk.document_id == args.document_id
                and chunk.document_version == args.product_version
                and chunk.chunk_id == args.chunk_id
            ):
                return {"found": True, "chunk": _chunk_to_dict(chunk)}
        return {"found": False, "chunk": None}


def _vector_to_pg_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class PgVectorHybridKnowledgeRetriever(KnowledgeRetrieverProtocol):
    """生产级混合检索器：PostgreSQL pgvector 向量召回 + BM25 关键词召回 + RRF 融合。"""

    def __init__(
        self,
        *,
        pool: AsyncConnectionPool[Any],
        embedder: EmbedderProtocol,
        rrf_k: int = 60,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._fault_injector = fault_injector
        self._bm25_cache: dict[str, tuple[list[KnowledgeChunkRecord], BM25Okapi]] = {}
        self._bm25_lock = asyncio.Lock()

    async def _load_bm25_for_version(
        self, product_version: str
    ) -> tuple[list[KnowledgeChunkRecord], BM25Okapi | None]:
        cached = self._bm25_cache.get(product_version)
        if cached is not None:
            return cached

        async with self._bm25_lock:
            cached = self._bm25_cache.get(product_version)
            if cached is not None:
                return cached

            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT document_id, document_version, chunk_id, title, content
                        FROM knowledge_chunks
                        WHERE document_version = %s
                        ORDER BY document_id, chunk_id
                        """,
                        (product_version,),
                    )
                    rows = await cur.fetchall()

            chunks = [
                KnowledgeChunkRecord(
                    document_id=row[0],
                    document_version=row[1],
                    chunk_id=row[2],
                    title=row[3],
                    text=row[4],
                )
                for row in rows
            ]
            if not chunks:
                return [], None

            corpus = [
                tokenize_preserving_identifiers(f"{c.title}\n{c.text}")
                for c in chunks
            ]
            bm25 = BM25Okapi(corpus)
            self._bm25_cache[product_version] = (chunks, bm25)
            return chunks, bm25

    def invalidate_bm25_cache(self) -> None:
        self._bm25_cache.clear()

    async def search_documents(self, args: SearchDocumentsArgs) -> dict[str, Any]:
        if self._fault_injector is not None:
            rule = await self._fault_injector.maybe_inject_tool_fault("search_documents")
            if rule is not None and rule.fault_type == "invalid_output_schema":
                return {"chunks": [{"text": "缺少来源字段的畸形返回体"}]}

        recall_depth = max(args.top_k * 2, 10)
        query_vec = await self._embedder.embed_query(args.query)
        vec_literal = _vector_to_pg_literal(query_vec)

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT document_id, document_version, chunk_id, title, content,
                           1 - (embedding <=> %s::vector) AS cosine_score
                    FROM knowledge_chunks
                    WHERE document_version = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec_literal, args.product_version, vec_literal, recall_depth),
                )
                vector_rows = await cur.fetchall()

        vector_ranked = [
            (
                KnowledgeChunkRecord(
                    document_id=r[0],
                    document_version=r[1],
                    chunk_id=r[2],
                    title=r[3],
                    text=r[4],
                ),
                float(r[5]),
            )
            for r in vector_rows
        ]

        version_chunks, bm25 = await self._load_bm25_for_version(args.product_version)
        bm25_ranked: list[tuple[KnowledgeChunkRecord, float]] = []
        if bm25 is not None and version_chunks:
            scores = bm25.get_scores(tokenize_preserving_identifiers(args.query))
            scored = [
                (version_chunks[idx], float(scores[idx]))
                for idx in range(len(version_chunks))
                if float(scores[idx]) > 0.0
            ]
            bm25_ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:recall_depth]

        fused = compute_rrf_fusion(vector_ranked, bm25_ranked, rrf_k=self._rrf_k)
        top_items = fused[: args.top_k]
        return {
            "chunks": [_chunk_to_dict(chunk, score) for chunk, score in top_items],
            "total_candidates": len(version_chunks),
        }

    async def read_document_chunk(self, args: ReadDocumentChunkArgs) -> dict[str, Any]:
        if self._fault_injector is not None:
            rule = await self._fault_injector.maybe_inject_tool_fault("read_document_chunk")
            if rule is not None and rule.fault_type == "invalid_output_schema":
                return {"found": True}

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT document_id, document_version, chunk_id, title, content
                    FROM knowledge_chunks
                    WHERE document_id = %s AND document_version = %s AND chunk_id = %s
                    LIMIT 1
                    """,
                    (args.document_id, args.product_version, args.chunk_id),
                )
                row = await cur.fetchone()

        if row is None:
            return {"found": False, "chunk": None}

        chunk = KnowledgeChunkRecord(
            document_id=row[0],
            document_version=row[1],
            chunk_id=row[2],
            title=row[3],
            text=row[4],
        )
        return {"found": True, "chunk": _chunk_to_dict(chunk)}
