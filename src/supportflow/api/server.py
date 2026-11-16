from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from supportflow.config import Settings, get_settings
from supportflow.deps import KnowledgeRetrieverProtocol, OperationsGatewayProtocol, RuntimeDeps, TicketStoreProtocol
from supportflow.env_factory import create_chat_model, create_embedder
from supportflow.graph import build_support_graph
from supportflow.nodes.supervisor import compute_budget_status
from supportflow.persistence.db import (
    create_pg_pool,
    create_postgres_checkpointer,
    init_database_schema,
)
from supportflow.persistence.ticket_service import (
    InMemoryTicketStore,
    PostgresTicketStore,
    _row_to_ticket_record,
)
from supportflow.schemas import FaultInjectorConfig, FaultRule, TicketRecord
from supportflow.state import create_initial_state
from supportflow.tools import FaultInjector
from supportflow.tools.knowledge import (
    DeterministicTestEmbedder,
    InMemoryHybridKnowledgeRetriever,
    PgVectorHybridKnowledgeRetriever,
)
from supportflow.tools.operations import SimulationOperationsGateway


# =============================================================================
# FastAPI 请求 / 响应模型
# =============================================================================


class FaultConfigRequest(BaseModel):
    rules: list[FaultRule] = Field(default_factory=list)
    reset_counters: bool = True


class CreateCaseRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str | None = None
    case_id: str | None = None
    account_id: str | None = None
    verified_account_id: str | None = None
    product_version: str | None = None
    request_ids: list[str] = Field(default_factory=list)
    fault_rules: list[FaultRule] | None = None


class ResumeCaseRequest(BaseModel):
    resume_payload: dict[str, Any] | str
    verified_account_id: str | None = None
    fault_rules: list[FaultRule] | None = None


class RetryTicketRequest(BaseModel):
    verified_account_id: str | None = None
    fault_rules: list[FaultRule] | None = None


class MessageView(BaseModel):
    role: str
    content: str


class CaseStateResponse(BaseModel):
    thread_id: str
    case_id: str
    status: Literal["running", "interrupted", "completed"]
    interrupts: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[MessageView] = Field(default_factory=list)
    problem: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    rounds: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    budget_status: dict[str, Any] = Field(default_factory=dict)
    ticket: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] | None = None


# =============================================================================
# 运行期共享容器与 Lifespan
# =============================================================================


@dataclass(slots=True)
class AppRuntimeContainer:
    settings: Settings
    graph: Any
    fault_injector: FaultInjector
    knowledge_retriever: KnowledgeRetrieverProtocol
    operations_gateway: OperationsGatewayProtocol
    ticket_store: TicketStoreProtocol
    llm: Any
    pool: AsyncConnectionPool[Any] | None = None
    active_fault_config: FaultInjectorConfig = field(default_factory=FaultInjectorConfig)
    thread_locks: dict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
    # 记录每个会话启动时的核验账户（避免 resume 时丢失 verified_account_id）
    session_accounts: dict[str, str | None] = field(default_factory=dict)

    def lock_for(self, thread_id: str) -> asyncio.Lock:
        return self.thread_locks[thread_id]

    def apply_fault_rules_if_changed(self, fault_rules: list[FaultRule] | None) -> None:
        if fault_rules is None:
            return
        new_cfg = FaultInjectorConfig(rules=list(fault_rules))
        if new_cfg != self.active_fault_config:
            self.active_fault_config = new_cfg
            self.fault_injector.update_config(new_cfg)

    def build_deps(
        self,
        *,
        verified_account_id: str | None = None,
    ) -> RuntimeDeps:
        return RuntimeDeps.build(
            settings=self.settings,
            knowledge_retriever=self.knowledge_retriever,
            operations_gateway=self.operations_gateway,
            ticket_store=self.ticket_store,
            supervisor_llm=self.llm,
            agent_llm=self.llm,
            verified_account_id=verified_account_id,
            fault_config=self.active_fault_config,
        )


def _serialize_snapshot(thread_id: str, snapshot: Any) -> CaseStateResponse:
    values: dict[str, Any] = dict(snapshot.values or {})
    if not values:
        raise HTTPException(status_code=404, detail=f"会话 thread_id={thread_id} 不存在")

    interrupts: list[dict[str, Any]] = []
    for task in snapshot.tasks or ():
        for intr in getattr(task, "interrupts", ()) or ():
            val = getattr(intr, "value", intr)
            if isinstance(val, dict):
                interrupts.append(val)
            else:
                interrupts.append({"payload": val})

    raw_msgs = values.get("messages") or []
    messages = [
        MessageView(role=type(m).__name__, content=str(getattr(m, "content", m)))
        for m in raw_msgs
    ]

    status: Literal["running", "interrupted", "completed"]
    if interrupts:
        status = "interrupted"
    elif snapshot.next:
        status = "running"
    else:
        status = "completed"

    return CaseStateResponse(
        thread_id=thread_id,
        case_id=str(values.get("case_id") or ""),
        status=status,
        interrupts=interrupts,
        messages=messages,
        problem=dict(values.get("problem") or {}),
        assessment=values.get("assessment"),
        decision=values.get("decision"),
        rounds=dict(values.get("rounds") or {}),
        evidence=dict(values.get("evidence") or {}),
        budget=dict(values.get("budget") or {}),
        budget_status=compute_budget_status(values),  # type: ignore[arg-type]
        ticket=dict(values.get("ticket") or {}),
        answer=values.get("answer"),
    )


def create_app(*, settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        fault_injector = FaultInjector(FaultInjectorConfig())
        llm = create_chat_model(resolved_settings)
        pool: AsyncConnectionPool[Any] | None = None

        if resolved_settings.runtime_mode == "live":
            pool = create_pg_pool(resolved_settings)
            await pool.open(wait=True)
            await init_database_schema(pool, embedding_dim=resolved_settings.embedding_dim)
            checkpointer = await create_postgres_checkpointer(pool)
            embedder = create_embedder(resolved_settings, use_test_embedder=False)
            retriever: KnowledgeRetrieverProtocol = PgVectorHybridKnowledgeRetriever(
                pool=pool,
                embedder=embedder,
                rrf_k=resolved_settings.rrf_k,
                fault_injector=fault_injector,
            )
            ticket_store: TicketStoreProtocol = PostgresTicketStore(
                pool=pool,
                fault_injector=fault_injector,
            )
        else:
            checkpointer = MemorySaver()
            retriever = InMemoryHybridKnowledgeRetriever(
                embedder=DeterministicTestEmbedder(dim=128),
                rrf_k=resolved_settings.rrf_k,
                fault_injector=fault_injector,
            )
            ticket_store = InMemoryTicketStore(fault_injector=fault_injector)

        gateway = SimulationOperationsGateway(fault_injector=fault_injector)
        graph = build_support_graph(checkpointer=checkpointer)

        app_instance.state.runtime_container = AppRuntimeContainer(
            settings=resolved_settings,
            graph=graph,
            fault_injector=fault_injector,
            knowledge_retriever=retriever,
            operations_gateway=gateway,
            ticket_store=ticket_store,
            llm=llm,
            pool=pool,
        )
        try:
            yield
        finally:
            if pool is not None:
                await pool.close()

    app_instance = FastAPI(
        title="SupportFlow API",
        version="0.1.0",
        description="基于 LangGraph 的多 Agent 技术支持与幂等工单工作台后端服务",
        lifespan=lifespan,
    )

    def _container() -> AppRuntimeContainer:
        return app_instance.state.runtime_container

    @app_instance.get("/health")
    async def health() -> dict[str, str]:
        c = _container()
        return {"status": "ok", "runtime_mode": c.settings.runtime_mode}

    @app_instance.get("/faults")
    async def get_faults() -> FaultInjectorConfig:
        return _container().active_fault_config

    @app_instance.post("/faults")
    async def set_faults(req: FaultConfigRequest) -> dict[str, Any]:
        c = _container()
        new_cfg = FaultInjectorConfig(rules=req.rules)
        c.active_fault_config = new_cfg
        if req.reset_counters:
            c.fault_injector.update_config(new_cfg)
        else:
            c.fault_injector._config = new_cfg
        return {
            "status": "updated",
            "rule_count": len(new_cfg.rules),
            "rules": [r.model_dump() for r in new_cfg.rules],
        }

    @app_instance.post("/cases", response_model=CaseStateResponse)
    async def create_case(req: CreateCaseRequest) -> CaseStateResponse:
        c = _container()
        thread_id = req.thread_id or f"thread-{uuid.uuid4().hex[:10]}"
        c.apply_fault_rules_if_changed(req.fault_rules)

        verified = req.verified_account_id
        c.session_accounts[thread_id] = verified
        effective_account = verified or req.account_id

        try:
            initial = create_initial_state(
                req.question,
                c.settings.to_budget_policy(),
                case_id=req.case_id,
                account_id=effective_account,
                product_version=req.product_version,
                request_ids=req.request_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        deps = c.build_deps(verified_account_id=verified)
        config = {"configurable": {"thread_id": thread_id}}

        async with c.lock_for(thread_id):
            await c.graph.ainvoke(initial, config=config, context=deps)
            snapshot = await c.graph.aget_state(config)

        return _serialize_snapshot(thread_id, snapshot)

    @app_instance.get("/cases/{thread_id}", response_model=CaseStateResponse)
    async def get_case(thread_id: str) -> CaseStateResponse:
        c = _container()
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await c.graph.aget_state(config)
        return _serialize_snapshot(thread_id, snapshot)

    @app_instance.post("/cases/{thread_id}/resume", response_model=CaseStateResponse)
    async def resume_case(thread_id: str, req: ResumeCaseRequest) -> CaseStateResponse:
        c = _container()
        c.apply_fault_rules_if_changed(req.fault_rules)
        verified = (
            req.verified_account_id
            if req.verified_account_id is not None
            else c.session_accounts.get(thread_id)
        )
        deps = c.build_deps(verified_account_id=verified)
        config = {"configurable": {"thread_id": thread_id}}

        async with c.lock_for(thread_id):
            existing = await c.graph.aget_state(config)
            if not (existing.values or {}):
                raise HTTPException(status_code=404, detail=f"会话 thread_id={thread_id} 不存在")
            try:
                await c.graph.ainvoke(
                    Command(resume=req.resume_payload),
                    config=config,
                    context=deps,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            snapshot = await c.graph.aget_state(config)

        return _serialize_snapshot(thread_id, snapshot)

    @app_instance.post("/cases/{thread_id}/ticket/retry", response_model=CaseStateResponse)
    async def retry_ticket_submission(
        thread_id: str,
        req: RetryTicketRequest | None = None,
    ) -> CaseStateResponse:
        c = _container()
        body = req or RetryTicketRequest()
        c.apply_fault_rules_if_changed(body.fault_rules)
        verified = (
            body.verified_account_id
            if body.verified_account_id is not None
            else c.session_accounts.get(thread_id)
        )
        deps = c.build_deps(verified_account_id=verified)
        config = {"configurable": {"thread_id": thread_id}}

        async with c.lock_for(thread_id):
            existing = await c.graph.aget_state(config)
            values = dict(existing.values or {})
            if not values:
                raise HTTPException(status_code=404, detail=f"会话 thread_id={thread_id} 不存在")

            ticket = values.get("ticket") or {}
            if ticket.get("submission") is None:
                raise HTTPException(
                    status_code=400,
                    detail="当前会话尚无已确认落盘的 TicketSubmission 快照，无法执行直达核对重试",
                )
            if ticket.get("status") == "cancelled":
                raise HTTPException(status_code=400, detail="已取消的工单流程不允许重试提交")

            await c.graph.ainvoke(
                {"entry_action": "retry_ticket_submission"},
                config=config,
                context=deps,
            )
            snapshot = await c.graph.aget_state(config)

        return _serialize_snapshot(thread_id, snapshot)

    @app_instance.get("/tickets", response_model=list[TicketRecord])
    async def list_tickets() -> list[TicketRecord]:
        c = _container()
        if isinstance(c.ticket_store, InMemoryTicketStore):
            return list(reversed(c.ticket_store.list_tickets()))
        if c.pool is not None:
            async with c.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT id, account_id, idempotency_key, payload_hash,
                               payload_json, title, status, created_at
                        FROM tickets
                        ORDER BY created_at DESC
                        LIMIT 50
                        """
                    )
                    rows = await cur.fetchall()
            return [_row_to_ticket_record(r) for r in rows]
        return []

    return app_instance


app = create_app()
