from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel

from supportflow.config import Settings, get_settings
from supportflow.schemas import (
    FaultInjectorConfig,
    ReadDocumentChunkArgs,
    SearchDocumentsArgs,
    TicketRecord,
    TicketSubmissionOutcome,
)
from supportflow.state import TicketSubmission


@runtime_checkable
class KnowledgeRetrieverProtocol(Protocol):
    """知识检索环境抽象接口（返回原始 dict 以便在 KV 节点执行严格的出参校验）。"""

    async def search_documents(self, args: SearchDocumentsArgs) -> dict[str, Any]:
        ...

    async def read_document_chunk(self, args: ReadDocumentChunkArgs) -> dict[str, Any]:
        ...


@runtime_checkable
class OperationsGatewayProtocol(Protocol):
    """只读业务诊断工具环境抽象接口（返回原始 dict 以便在 OV 节点执行严格出参校验）。"""

    async def invoke_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        ...


@runtime_checkable
class TicketStoreProtocol(Protocol):
    """工单持久化与幂等事务抽象接口。"""

    async def find_by_idempotency_key(
        self,
        account_id: str,
        idempotency_key: str,
    ) -> TicketRecord | None:
        ...

    async def create_idempotent(
        self,
        submission: TicketSubmission,
        account_id: str,
    ) -> TicketSubmissionOutcome:
        ...


@runtime_checkable
class ClockProtocol(Protocol):
    """系统时钟抽象接口，便于耗时统计与测试时间控制。"""

    def monotonic(self) -> float:
        ...

    def now_iso(self) -> str:
        ...


class SystemClock:
    """默认系统单调时钟与 UTC ISO-8601 时钟。"""

    def monotonic(self) -> float:
        return time.monotonic()

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RuntimeDeps:
    """LangGraph context_schema 运行时依赖注入容器。

    说明：所有不可序列化的连接池、客户端句柄、时钟、仿真故障配置都放在这里，
    通过 graph.ainvoke(..., context=deps) 注入，绝不污染 SupportState Checkpoint。
    """

    settings: Settings
    knowledge_retriever: KnowledgeRetrieverProtocol
    operations_gateway: OperationsGatewayProtocol
    ticket_store: TicketStoreProtocol
    supervisor_llm: BaseChatModel
    agent_llm: BaseChatModel
    clock: ClockProtocol = field(default_factory=SystemClock)
    # 单租户能力沙箱边界：由会话上下文注入已核验账户 ID，防止模型越权查询其他账户
    verified_account_id: str | None = None
    fault_config: FaultInjectorConfig = field(default_factory=FaultInjectorConfig)

    @classmethod
    def build(
        cls,
        *,
        knowledge_retriever: KnowledgeRetrieverProtocol,
        operations_gateway: OperationsGatewayProtocol,
        ticket_store: TicketStoreProtocol,
        supervisor_llm: BaseChatModel,
        agent_llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        clock: ClockProtocol | None = None,
        verified_account_id: str | None = None,
        fault_config: FaultInjectorConfig | None = None,
    ) -> RuntimeDeps:
        resolved_settings = settings or get_settings()
        return cls(
            settings=resolved_settings,
            knowledge_retriever=knowledge_retriever,
            operations_gateway=operations_gateway,
            ticket_store=ticket_store,
            supervisor_llm=supervisor_llm,
            agent_llm=agent_llm or supervisor_llm,
            clock=clock or SystemClock(),
            verified_account_id=verified_account_id,
            fault_config=fault_config or FaultInjectorConfig(),
        )
