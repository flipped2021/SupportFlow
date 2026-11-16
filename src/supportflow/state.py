from __future__ import annotations

import hashlib
import json
import uuid
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

AgentName = Literal["knowledge", "operations"]


class ProblemContext(TypedDict):
    original_question: str
    account_id: str | None
    product_version: str | None
    error_codes: list[str]
    request_ids: list[str]
    known_facts: dict[str, str]
    attempted_steps: list[str]


class EvidenceConflict(TypedDict):
    description: str
    evidence_ids: list[str]


class Assessment(TypedDict):
    summary: str
    evidence_sufficient: bool
    information_gaps: list[str]
    conflicts: list[EvidenceConflict]
    # 跨轮次工具健康度感知：记录上一轮出现服务端异常或契约损坏的工具提示
    tool_health_notes: list[str]


class Decision(TypedDict):
    action: Literal["investigate", "ask_user", "answer", "draft_ticket"]
    reason: str


class BudgetPolicy(TypedDict):
    max_rounds: int
    max_tool_attempts: int
    max_investigation_ms: int
    max_agent_steps_per_task: int


class TaskPlan(TypedDict):
    task_id: str
    agent: AgentName
    objective: str
    max_tool_attempts: int
    max_steps: int
    timeout_ms: int


class RoundPlan(TypedDict):
    round_id: str
    tasks: list[TaskPlan]
    timeout_ms: int


class Usage(TypedDict):
    tool_attempts: int
    agent_steps: int
    elapsed_ms: int


class ExecutionError(TypedDict):
    kind: Literal[
        "argument_validation",
        "output_validation",
        "timeout",
        "connection",
        "tool_error",
        "unhandled",
    ]
    message: str
    tool_name: str | None
    logical_call_id: str | None


class DocumentSource(TypedDict):
    kind: Literal["document"]
    document_id: str
    document_version: str
    chunk_id: str


class BusinessSource(TypedDict):
    kind: Literal["business"]
    tool_name: str
    logical_call_id: str
    query_params: dict[str, Any]
    queried_at: str
    outcome: Literal["normal", "business_error", "no_data"]


class Evidence(TypedDict):
    evidence_id: str
    content: str
    source: DocumentSource | BusinessSource


class DiagnosisResult(TypedDict):
    round_id: str
    task_id: str
    agent: AgentName
    status: Literal[
        "completed",
        "no_answer",
        "budget_exhausted",
        "timeout",
        "failed",
    ]
    summary: str
    evidence: list[Evidence]
    errors: list[ExecutionError]
    usage: Usage


class InvestigationRound(TypedDict):
    plan: RoundPlan
    results: list[DiagnosisResult] | None
    elapsed_ms: int | None


class Answer(TypedDict):
    content: str
    citation_evidence_ids: list[str]
    execution_summary: str
    uncertainties: list[str]


class TicketDraft(TypedDict):
    draft_id: str
    version: int
    title: str
    description: str
    evidence_ids: list[str]
    attempted_steps: list[str]


class TicketReview(TypedDict):
    draft_id: str
    draft_version: int
    action: Literal["approve", "edit", "cancel"]
    edit_request: str | None


class TicketSubmission(TypedDict):
    idempotency_key: str
    approved_draft_id: str
    approved_draft_version: int
    payload_json: str
    payload_hash: str


class TicketState(TypedDict):
    status: Literal[
        "not_started",
        "draft",
        "approved",
        "prepared",
        "succeeded",
        "cancelled",
        "conflict",
        "failed",
        "unknown",
    ]
    draft: TicketDraft | None
    review: TicketReview | None
    submission: TicketSubmission | None
    ticket_id: str | None
    result_kind: Literal["created", "existing"] | None
    error_message: str | None


class SupportState(TypedDict):
    """主图持久化 State（由 PostgreSQL checkpointer 持久化）。"""

    case_id: str
    # 用于入口路由区分：常规诊断 vs 对 unknown 工单发起核对/重试（直达 execute_submission）
    entry_action: Literal["diagnose", "retry_ticket_submission"]
    messages: Annotated[list[AnyMessage], add_messages]
    problem: ProblemContext
    assessment: Assessment | None
    decision: Decision | None
    clarification_questions: list[str]
    current_round_id: str | None
    rounds: dict[str, InvestigationRound]
    evidence: dict[str, Evidence]
    budget: BudgetPolicy
    ticket: TicketState
    answer: Answer | None


class LogicalCallRecord(TypedDict):
    logical_call_id: str
    tool_name: str
    args: dict[str, Any]
    attempts_started: int
    retry_count: int
    retry_authorized: bool
    last_error: ExecutionError | None


class AgentLocalState(TypedDict):
    """子 Agent 运行期局部状态（由包装器持有，不混入主图 State）。"""

    round_id: str
    task: TaskPlan
    messages: list[AnyMessage]
    logical_calls: dict[str, LogicalCallRecord]
    evidence: dict[str, Evidence]
    errors: list[ExecutionError]
    usage: Usage
    # output_validation 触发的本轮局部熔断工具集合: tool_name -> reason
    disabled_tools: dict[str, str]
    # 连续无新证据产出的步数计数，达到 2 时提前收敛退出
    consecutive_no_evidence_steps: int
    deadline_monotonic: float


def empty_ticket_state() -> TicketState:
    return {
        "status": "not_started",
        "draft": None,
        "review": None,
        "submission": None,
        "ticket_id": None,
        "result_kind": None,
        "error_message": None,
    }


def create_initial_state(
    question: str,
    budget: BudgetPolicy,
    *,
    case_id: str | None = None,
    account_id: str | None = None,
    product_version: str | None = None,
    error_codes: list[str] | None = None,
    request_ids: list[str] | None = None,
    known_facts: dict[str, str] | None = None,
    attempted_steps: list[str] | None = None,
) -> SupportState:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question must not be empty")

    resolved_case_id = case_id or f"case-{uuid.uuid4().hex[:12]}"
    problem: ProblemContext = {
        "original_question": cleaned_question,
        "account_id": account_id,
        "product_version": product_version,
        "error_codes": list(error_codes or []),
        "request_ids": list(request_ids or []),
        "known_facts": dict(known_facts or {}),
        "attempted_steps": list(attempted_steps or []),
    }

    return {
        "case_id": resolved_case_id,
        "entry_action": "diagnose",
        "messages": [HumanMessage(content=cleaned_question)],
        "problem": problem,
        "assessment": None,
        "decision": None,
        "clarification_questions": [],
        "current_round_id": None,
        "rounds": {},
        "evidence": {},
        "budget": dict(budget),
        "ticket": empty_ticket_state(),
        "answer": None,
    }


def create_agent_local_state(
    *,
    round_id: str,
    task: TaskPlan,
    initial_messages: list[AnyMessage],
    deadline_monotonic: float,
) -> AgentLocalState:
    return {
        "round_id": round_id,
        "task": task,
        "messages": list(initial_messages),
        "logical_calls": {},
        "evidence": {},
        "errors": [],
        "usage": {
            "tool_attempts": 0,
            "agent_steps": 0,
            "elapsed_ms": 0,
        },
        "disabled_tools": {},
        "consecutive_no_evidence_steps": 0,
        "deadline_monotonic": deadline_monotonic,
    }


def make_logical_call_id(
    task_id: str,
    tool_name: str,
    validated_args: dict[str, Any],
) -> str:
    """按任务 ID + 工具名 + 规范化参数生成稳定的逻辑调用 ID，防止跨循环重置重试计数。"""
    canonical = json.dumps(
        {"task_id": task_id, "tool": tool_name, "args": validated_args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"call-{digest}"
