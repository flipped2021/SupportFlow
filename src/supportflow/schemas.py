from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from supportflow.state import AgentName

# =============================================================================
# 跨边界显式异常定义（供工具层、事务服务层与编排层精准分流）
# =============================================================================


class SupportFlowError(Exception):
    """项目基础异常。"""


class TransientToolError(SupportFlowError):
    """临时超时或网络连接异常（运行诊断允许对同一逻辑调用最多重试一次）。"""

    def __init__(self, message: str, *, kind: Literal["timeout", "connection"] = "timeout"):
        super().__init__(message)
        self.kind = kind


class DefiniteToolError(SupportFlowError):
    """确定性工具调用失败（非瞬时故障，不触发重试）。"""


class CommitOutcomeUnknownError(SupportFlowError):
    """数据库事务已发送提交但客户端未收到确认，结果待核实（对应工单 UNKNOWN 分支）。"""


class DefiniteDatabaseError(SupportFlowError):
    """数据库已明确回滚或拒绝的确定性失败（对应工单 ERROR 分支）。"""


# =============================================================================
# 知识诊断工具：入参 / 出参 Pydantic 严格校验模型
# =============================================================================


class SearchDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=500, description="检索关键词或自然语言问题描述")
    product_version: str = Field(min_length=1, max_length=32, description="目标产品版本号，例如 v2.3")
    top_k: int = Field(default=5, ge=1, le=15, description="返回分块条数上限")

    @field_validator("query", "product_version")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class ReadDocumentChunkArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: str = Field(min_length=1, max_length=128)
    product_version: str = Field(min_length=1, max_length=32)
    chunk_id: str = Field(min_length=1, max_length=128)

    @field_validator("document_id", "product_version", "chunk_id")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class RetrievedChunkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float | None = None


class SearchDocumentsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[RetrievedChunkModel]
    total_candidates: int = Field(ge=0)


class ReadDocumentChunkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: RetrievedChunkModel | None = None
    found: bool = True

    @model_validator(mode="after")
    def _validate_presence(self) -> ReadDocumentChunkOutput:
        if self.found and self.chunk is None:
            raise ValueError("chunk must be present when found is True")
        return self


# =============================================================================
# 运行诊断 4 类只读业务工具：入参 / 出参 Pydantic 严格校验模型
# =============================================================================


def _validate_iso8601(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("timestamp must not be blank")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone information (e.g. Z or +08:00)")
    return cleaned


class GetAccountPermissionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    account_id: str = Field(min_length=1, max_length=64)

    @field_validator("account_id")
    @classmethod
    def _strip_account_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("account_id must not be blank")
        return cleaned


class GetAccountPermissionsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    status: Literal["active", "suspended", "restricted"]
    permissions: list[str]
    token_scopes: list[str]
    can_upload: bool
    checked_at: str

    @field_validator("checked_at")
    @classmethod
    def _check_ts(cls, value: str) -> str:
        checked = _validate_iso8601(value)
        assert checked is not None
        return checked


class GetAccountQuotaArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    account_id: str = Field(min_length=1, max_length=64)

    @field_validator("account_id")
    @classmethod
    def _strip_account_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("account_id must not be blank")
        return cleaned


class GetAccountQuotaOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1)
    storage_used_bytes: int = Field(ge=0)
    storage_limit_bytes: int = Field(gt=0)
    daily_api_calls: int = Field(ge=0)
    daily_api_limit: int = Field(gt=0)
    quota_exceeded: bool
    checked_at: str

    @field_validator("checked_at")
    @classmethod
    def _check_ts(cls, value: str) -> str:
        checked = _validate_iso8601(value)
        assert checked is not None
        return checked


class GetServiceStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    service_name: Literal["upload", "auth", "storage", "gateway", "all"]
    region: str = Field(default="cn-south-1", min_length=1, max_length=32)


class GetServiceStatusOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str = Field(min_length=1)
    region: str = Field(min_length=1)
    health: Literal["healthy", "degraded", "outage"]
    incident_id: str | None = None
    summary: str = Field(min_length=1)
    checked_at: str

    @field_validator("checked_at")
    @classmethod
    def _check_ts(cls, value: str) -> str:
        checked = _validate_iso8601(value)
        assert checked is not None
        return checked


class GetRequestLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    account_id: str = Field(min_length=1, max_length=64)
    request_id: str | None = Field(default=None, min_length=1, max_length=64)
    start_time: str | None = None
    end_time: str | None = None
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("account_id")
    @classmethod
    def _strip_account_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("account_id must not be blank")
        return cleaned

    @field_validator("start_time", "end_time")
    @classmethod
    def _validate_timestamps(cls, value: str | None) -> str | None:
        return _validate_iso8601(value)

    @model_validator(mode="after")
    def _validate_range_and_filters(self) -> GetRequestLogsArgs:
        if self.start_time is not None and self.end_time is not None:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            if start_dt > end_dt:
                raise ValueError("start_time must be earlier than or equal to end_time")
        return self


class RequestLogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    timestamp: str
    endpoint: str = Field(min_length=1)
    http_status: int = Field(ge=100, le=599)
    error_code: str | None = None
    reason: str = Field(min_length=1)
    product_version: str = Field(min_length=1)


class GetRequestLogsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1)
    entries: list[RequestLogEntry]
    checked_at: str

    @field_validator("checked_at")
    @classmethod
    def _check_ts(cls, value: str) -> str:
        checked = _validate_iso8601(value)
        assert checked is not None
        return checked


TOOL_INPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "search_documents": SearchDocumentsArgs,
    "read_document_chunk": ReadDocumentChunkArgs,
    "get_account_permissions": GetAccountPermissionsArgs,
    "get_account_quota": GetAccountQuotaArgs,
    "get_service_status": GetServiceStatusArgs,
    "get_request_logs": GetRequestLogsArgs,
}

TOOL_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "search_documents": SearchDocumentsOutput,
    "read_document_chunk": ReadDocumentChunkOutput,
    "get_account_permissions": GetAccountPermissionsOutput,
    "get_account_quota": GetAccountQuotaOutput,
    "get_service_status": GetServiceStatusOutput,
    "get_request_logs": GetRequestLogsOutput,
}


# =============================================================================
# LLM 结构化输出模型（Supervisor、Plan、Answer、Draft、SubAgent 收尾）
# =============================================================================


class ConflictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class SupervisorAnalysisOutput(BaseModel):
    """Supervisor 分析问题、合并证据与决策时的结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    account_id: str | None = None
    product_version: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    request_ids: list[str] = Field(default_factory=list)
    known_facts_updates: dict[str, str] = Field(default_factory=dict)

    summary: str = Field(min_length=1)
    evidence_sufficient: bool
    information_gaps: list[str] = Field(default_factory=list)
    conflicts: list[ConflictModel] = Field(default_factory=list)

    requires_user_input: bool = False
    clarification_questions: list[str] = Field(default_factory=list)

    recommended_action: Literal["investigate", "ask_user", "answer", "draft_ticket"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_clarification_consistency(self) -> SupervisorAnalysisOutput:
        if self.recommended_action == "ask_user" and not self.clarification_questions:
            raise ValueError("clarification_questions must not be empty when asking user")
        return self


class PlannedTaskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    objective: str = Field(min_length=5, max_length=500)
    max_tool_attempts: int = Field(default=3, ge=1, le=10)
    max_steps: int = Field(default=5, ge=1, le=12)
    timeout_ms: int = Field(default=15000, ge=1000, le=60000)


class InvestigationPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[PlannedTaskOutput] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _validate_distinct_agents(self) -> InvestigationPlanOutput:
        agents = [t.agent for t in self.tasks]
        if len(agents) != len(set(agents)):
            raise ValueError("each agent can appear at most once per round")
        return self


class AnswerCompositionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    citation_evidence_ids: list[str] = Field(default_factory=list)
    execution_summary: str = Field(min_length=1)
    uncertainties: list[str] = Field(default_factory=list)


class DraftTicketOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=10)
    evidence_ids: list[str] = Field(default_factory=list)
    attempted_steps: list[str] = Field(default_factory=list)


class AgentFinalSummaryOutput(BaseModel):
    """子 Agent 完成工具调用后产出的结构化结论。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "no_answer"]
    summary: str = Field(min_length=1)
    cited_evidence_ids: list[str] = Field(default_factory=list)


# =============================================================================
# 人工介入（HITL）恢复载荷校验模型
# =============================================================================


class ClarificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    account_id: str | None = None
    product_version: str | None = None
    request_ids: list[str] = Field(default_factory=list)
    extra_facts: dict[str, str] = Field(default_factory=dict)


class TicketReviewAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str = Field(min_length=1)
    draft_version: int = Field(ge=1)
    action: Literal["approve", "edit", "cancel"]
    edit_request: str | None = None

    @model_validator(mode="after")
    def _validate_edit_request(self) -> TicketReviewAction:
        if self.action == "edit":
            if self.edit_request is None or not self.edit_request.strip():
                raise ValueError("edit_request is required when action is 'edit'")
        return self


# =============================================================================
# 工单持久化领域模型与故障注入规则模型
# =============================================================================


class TicketPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    attempted_steps: list[str] = Field(default_factory=list)
    product_version: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    request_ids: list[str] = Field(default_factory=list)


class TicketRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    payload_hash: str = Field(min_length=1)
    payload_json: str = Field(min_length=1)
    title: str = Field(min_length=1)
    status: str = Field(default="open")
    created_at: str = Field(min_length=1)


class TicketSubmissionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["created", "existing", "conflict", "failed", "unknown"]
    ticket: TicketRecord | None = None
    error_message: str | None = None


FaultType = Literal[
    "timeout",
    "connection_error",
    "invalid_output_schema",
    "definite_tool_error",
    "db_commit_unknown",
    "db_definite_error",
]


class FaultRule(BaseModel):
    """可控故障注入规则，用于测试RETRY/MARK、WRAP超时、契约熔断及工单UNKNOWN恢复。"""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1, description="工具名或 'ticket_commit'")
    fault_type: FaultType
    trigger_on_attempt: int | None = Field(
        default=None,
        ge=1,
        description="仅在第 N 次调用尝试时触发；None 表示每次尝试均触发",
    )
    delay_ms: int = Field(default=0, ge=0, le=60000)
    detail: str = Field(default="injected fault for testing")


class FaultInjectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[FaultRule] = Field(default_factory=list)
