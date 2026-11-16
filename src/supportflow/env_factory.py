from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any, Callable, Literal, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from supportflow.config import Settings, get_settings
from supportflow.deps import KnowledgeRetrieverProtocol, OperationsGatewayProtocol, RuntimeDeps, TicketStoreProtocol
from supportflow.persistence.ticket_service import InMemoryTicketStore, PostgresTicketStore
from supportflow.schemas import (
    AnswerCompositionOutput,
    ConflictModel,
    DraftTicketOutput,
    FaultInjectorConfig,
    InvestigationPlanOutput,
    PlannedTaskOutput,
    SupervisorAnalysisOutput,
)
from supportflow.tools import FaultInjector
from supportflow.tools.knowledge import (
    DeterministicTestEmbedder,
    EmbedderProtocol,
    InMemoryHybridKnowledgeRetriever,
    OpenAICompatibleEmbedder,
    PgVectorHybridKnowledgeRetriever,
    SentenceTransformerEmbedder,
)
from supportflow.tools.operations import SimulationOperationsGateway

_ACC_RE = re.compile(r"\b(acc-[A-Za-z0-9_-]+)\b")
_REQ_RE = re.compile(r"\b(req-[A-Za-z0-9_-]+)\b")
_VER_RE = re.compile(r"\b(v\d+\.\d+(?:\.\d+)?)\b")
_ERR_RE = re.compile(r"\b(E_[A-Z0-9_]+)\b")
_HTTP_RE = re.compile(r"\b(401|403|429|500|502|503)\b")
_EVIDENCE_ID_RE = re.compile(r"\[(doc:[^\]]+|biz:[^\]]+)\]")


def _extract_tagged_block(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}[^>]*>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return match.group(1) if match else None


def _extract_tagged_json(text: str, tag: str) -> dict[str, Any] | None:
    raw = _extract_tagged_block(text, tag)
    if raw is None:
        return None
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else None
    except ValueError:
        return None


def _to_message_list(prompt_input: Any) -> list[BaseMessage]:
    if hasattr(prompt_input, "to_messages"):
        return list(prompt_input.to_messages())
    if isinstance(prompt_input, Sequence) and not isinstance(prompt_input, (str, bytes)):
        return list(prompt_input)
    raise TypeError(f"Unsupported prompt input type: {type(prompt_input)}")


class SimulationChatModel(BaseChatModel):
    """支持 bind_tools 与 with_structured_output 的离线确定性规则驱动仿真模型。

    通过无状态检查历史消息实现并发安全，供本地开发、60 条对比评测及自动化测试使用。
    """

    bound_tool_names: tuple[str, ...] = ()
    wrong_account_first_call: str | None = None
    force_repeat_tool_call: str | None = None
    omit_uncertainties_in_llm: bool = False
    model_name: str = "simulation-rule-llm"

    @property
    def _llm_type(self) -> str:
        return "simulation-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        del tool_choice, kwargs
        names: list[str] = []
        for item in tools:
            if isinstance(item, dict):
                fn = item.get("function") or {}
                name = fn.get("name") or item.get("name")
                if name:
                    names.append(str(name))
            elif hasattr(item, "name"):
                names.append(str(item.name))
            elif isinstance(item, type) and issubclass(item, BaseModel):
                names.append(item.__name__)
        return self.model_copy(update={"bound_tool_names": tuple(names)})

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        del kwargs
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError("SimulationChatModel.with_structured_output 仅支持 Pydantic BaseModel")

        def _sync_call(prompt_input: Any) -> BaseModel:
            messages = _to_message_list(prompt_input)
            return self._produce_structured(schema, messages)

        async def _async_call(prompt_input: Any) -> BaseModel:
            messages = _to_message_list(prompt_input)
            return self._produce_structured(schema, messages)

        return RunnableLambda(func=_sync_call, afunc=_async_call)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        ai_message = self._produce_tool_or_summary_message(messages)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        ai_message = self._produce_tool_or_summary_message(messages)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ------------------------------------------------------------------
    # 子 Agent 工具决策推理逻辑
    # ------------------------------------------------------------------
    def _produce_tool_or_summary_message(self, messages: list[BaseMessage]) -> AIMessage:
        human_text = str(messages[1].content) if len(messages) > 1 else str(messages[-1].content)
        problem = _extract_tagged_json(human_text, "problem_context") or {}
        known_evidence_text = _extract_tagged_block(human_text, "known_evidence") or ""

        question = str(problem.get("original_question") or "")
        account_id = problem.get("account_id")
        product_version = problem.get("product_version") or "v2.3"
        request_ids = list(problem.get("request_ids") or [])
        error_codes = list(problem.get("error_codes") or [])

        # 统计本轮子 Agent 已经执行过的工具情况与已产出的证据 ID
        called_tools: list[str] = []
        called_with_args: list[tuple[str, dict[str, Any]]] = []
        tool_errors: dict[str, list[str]] = {}
        observed_evidence_ids: list[str] = []
        observed_payloads: dict[str, dict[str, Any]] = {}

        call_id_to_info: dict[str, tuple[str, dict[str, Any]]] = {}
        for msg in messages[2:]:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", []) or []:
                    tname = str(tc["name"])
                    targs = dict(tc.get("args") or {})
                    cid = str(tc.get("id") or "")
                    called_tools.append(tname)
                    called_with_args.append((tname, targs))
                    if cid:
                        call_id_to_info[cid] = (tname, targs)
            elif isinstance(msg, ToolMessage):
                text = str(msg.content)
                tname, _ = call_id_to_info.get(msg.tool_call_id, ("unknown", {}))
                obs_body = _extract_tagged_block(text, "tool_observation")
                if obs_body is not None:
                    first_line, _, rest = obs_body.partition("\n")
                    if first_line.startswith("recorded_evidence_ids="):
                        try:
                            ids = ast.literal_eval(first_line.split("=", 1)[1].strip())
                            if isinstance(ids, list):
                                observed_evidence_ids.extend(str(i) for i in ids)
                        except (ValueError, SyntaxError):
                            pass
                    try:
                        observed_payloads[tname] = json.loads(rest)
                    except ValueError:
                        pass
                else:
                    tool_errors.setdefault(tname, []).append(text)

        available = set(self.bound_tool_names)

        # 测试钩子 1：首次调用故意传入越权账户，测试 OP 租户沙箱拦截与自纠
        if (
            self.wrong_account_first_call
            and "get_account_permissions" in available
            and not any(
                t == "get_account_permissions"
                and a.get("account_id") == self.wrong_account_first_call
                for t, a in called_with_args
            )
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"tc-{uuid.uuid4().hex[:8]}",
                        "name": "get_account_permissions",
                        "args": {"account_id": self.wrong_account_first_call},
                        "type": "tool_call",
                    }
                ],
            )

        # 测试钩子 2：在已有一次失败后，故意跨循环重放完全相同的工具+参数，测试逻辑重试上限拦截
        if (
            self.force_repeat_tool_call
            and self.force_repeat_tool_call in available
            and called_tools.count(self.force_repeat_tool_call) == 1
        ):
            prev_args = next(
                (a for t, a in called_with_args if t == self.force_repeat_tool_call),
                {"account_id": account_id or "acc-001"},
            )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"tc-{uuid.uuid4().hex[:8]}",
                        "name": self.force_repeat_tool_call,
                        "args": prev_args,
                        "type": "tool_call",
                    }
                ],
            )

        # 1) 知识检索策略
        if "search_documents" in available:
            search_count = called_tools.count("search_documents")
            if search_count == 0:
                q_parts = [question]
                if error_codes:
                    q_parts.extend(error_codes)
                if "403" in question and "storage:write" not in question:
                    q_parts.append("上传 403 E_UPLOAD_403 storage:write")
                query = " ".join(q_parts)[:220]
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"tc-{uuid.uuid4().hex[:8]}",
                            "name": "search_documents",
                            "args": {
                                "query": query,
                                "product_version": product_version,
                                "top_k": 5,
                            },
                            "type": "tool_call",
                        }
                    ],
                )
            # 若搜索未找到任何文档（例如测试无答案版本 v9.9），尝试换关键词再查 1 次以配合 2 步无进展退出规则
            search_obs = observed_payloads.get("search_documents", {})
            if (
                search_count == 1
                and isinstance(search_obs, dict)
                and len(search_obs.get("chunks", [])) == 0
            ):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"tc-{uuid.uuid4().hex[:8]}",
                            "name": "search_documents",
                            "args": {
                                "query": "E_UPLOAD_403 上传权限",
                                "product_version": product_version,
                                "top_k": 5,
                            },
                            "type": "tool_call",
                        }
                    ],
                )

        if (
            "read_document_chunk" in available
            and "read_document_chunk" not in called_tools
            and any(kw in question for kw in ("升级", "兼容", "v2.2", "变化"))
            and "doc:upload-auth-guide:v2.3:chunk-02" not in observed_evidence_ids
            and "doc:upload-auth-guide:v2.3:chunk-02" not in known_evidence_text
            and product_version == "v2.3"
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"tc-{uuid.uuid4().hex[:8]}",
                        "name": "read_document_chunk",
                        "args": {
                            "document_id": "upload-auth-guide",
                            "product_version": "v2.3",
                            "chunk_id": "chunk-02",
                        },
                        "type": "tool_call",
                    }
                ],
            )

        # 2) 运行诊断业务工具策略
        if account_id:
            need_perm = any(
                kw in question.lower()
                for kw in ("403", "上传", "权限", "upload", "token", "冻结", "升级", "e_upload", "e_account")
            )
            perm_attempted_valid = any(
                t == "get_account_permissions" and a.get("account_id") == account_id
                for t, a in called_with_args
            )
            if "get_account_permissions" in available and need_perm and not perm_attempted_valid:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"tc-{uuid.uuid4().hex[:8]}",
                            "name": "get_account_permissions",
                            "args": {"account_id": account_id},
                            "type": "tool_call",
                        }
                    ],
                )

            if (
                "get_request_logs" in available
                and request_ids
                and "get_request_logs" not in called_tools
            ):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"tc-{uuid.uuid4().hex[:8]}",
                            "name": "get_request_logs",
                            "args": {
                                "account_id": account_id,
                                "request_id": request_ids[0],
                                "limit": 10,
                            },
                            "type": "tool_call",
                        }
                    ],
                )

            perm_obs = observed_payloads.get("get_account_permissions")
            perm_ok = isinstance(perm_obs, dict) and perm_obs.get("can_upload") is True
            need_quota = (
                any(kw in question.lower() for kw in ("配额", "空间", "容量", "429", "quota", "限制"))
                or perm_ok
            )
            if (
                "get_account_quota" in available
                and need_quota
                and "get_account_quota" not in called_tools
            ):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"tc-{uuid.uuid4().hex[:8]}",
                            "name": "get_account_quota",
                            "args": {"account_id": account_id},
                            "type": "tool_call",
                        }
                    ],
                )

            quota_obs = observed_payloads.get("get_account_quota")
            quota_ok = isinstance(quota_obs, dict) and quota_obs.get("quota_exceeded") is False
            need_svc = (
                any(
                    kw in question.lower()
                    for kw in ("503", "服务", "网关", "region", "cn-east-1", "降级", "超时", "multipart")
                )
                or (perm_ok and quota_ok)
            )
            if (
                "get_service_status" in available
                and need_svc
                and "get_service_status" not in called_tools
            ):
                region = "cn-east-1" if "cn-east-1" in question else "cn-south-1"
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"tc-{uuid.uuid4().hex[:8]}",
                            "name": "get_service_status",
                            "args": {"service_name": "upload", "region": region},
                            "type": "tool_call",
                        }
                    ],
                )

        # 3) 结束调查，输出 JSON 总结
        if observed_evidence_ids:
            parts = []
            for tname, payload in observed_payloads.items():
                if tname == "search_documents":
                    chunks = payload.get("chunks") or []
                    if chunks:
                        parts.append(f"检索命中 {len(chunks)} 条版本文档（{chunks[0].get('title', '')}）")
                elif tname == "read_document_chunk" and payload.get("chunk"):
                    parts.append(f"补充读取文档分块 {payload['chunk'].get('chunk_id')}")
                elif tname == "get_account_permissions":
                    parts.append(
                        f"账户权限核查: status={payload.get('status')}, "
                        f"can_upload={payload.get('can_upload')}, scopes={payload.get('token_scopes')}"
                    )
                elif tname == "get_account_quota":
                    parts.append(f"账户配额核查: quota_exceeded={payload.get('quota_exceeded')}")
                elif tname == "get_service_status":
                    parts.append(f"服务状态核查: health={payload.get('health')}")
                elif tname == "get_request_logs":
                    entries = payload.get("entries") or []
                    if entries:
                        e0 = entries[0]
                        parts.append(
                            f"请求日志命中 {e0.get('request_id')}: HTTP {e0.get('http_status')} ({e0.get('error_code')})"
                        )
                    else:
                        parts.append("指定请求日志无命中记录")
            summary = "；".join(parts) if parts else "已完成本轮工具调查并收集证据"
            final_payload = {
                "status": "completed",
                "summary": summary,
                "cited_evidence_ids": observed_evidence_ids,
            }
        else:
            final_payload = {
                "status": "no_answer",
                "summary": "未能在知识库或可用业务工具中检索到有效支撑证据。",
                "cited_evidence_ids": [],
            }
        return AIMessage(content=json.dumps(final_payload, ensure_ascii=False))

    # ------------------------------------------------------------------
    # 主图 Supervisor / Plan / Answer / Ticket 结构化输出推理逻辑
    # ------------------------------------------------------------------
    def _produce_structured(
        self,
        schema: type[BaseModel],
        messages: list[BaseMessage],
    ) -> BaseModel:
        human_text = str(messages[-1].content)

        if schema is SupervisorAnalysisOutput:
            sup_view = _extract_tagged_json(human_text, "supervisor_view")
            if sup_view is not None:
                return self._analyze_in_sup(sup_view)
            merge_view = _extract_tagged_json(human_text, "merge_view")
            if merge_view is not None:
                return self._analyze_in_merge(merge_view)
            return self._analyze_in_sup({})

        if schema is InvestigationPlanOutput:
            plan_view = _extract_tagged_json(human_text, "plan_context") or {}
            return self._build_plan(plan_view)

        if schema is AnswerCompositionOutput:
            answer_view = _extract_tagged_json(human_text, "answer_context") or {}
            return self._compose_answer(answer_view)

        if schema is DraftTicketOutput:
            draft_view = _extract_tagged_json(human_text, "draft_context")
            if draft_view is not None:
                return self._draft_ticket(draft_view)
            old_draft = _extract_tagged_json(human_text, "old_draft") or {}
            edit_req = _extract_tagged_block(human_text, "edit_request") or ""
            return self._edit_ticket(old_draft, edit_req)

        raise ValueError(f"Unsupported structured schema: {schema.__name__}")

    def _extract_entities(
        self,
        problem: dict[str, Any],
        dialogue: str,
    ) -> tuple[str | None, str | None, list[str], list[str]]:
        combined = f"{problem.get('original_question', '')}\n{dialogue}"
        acc = problem.get("account_id")
        if not acc:
            m = _ACC_RE.search(combined)
            acc = m.group(1) if m else None
        ver = problem.get("product_version")
        if not ver:
            m = _VER_RE.search(combined)
            ver = m.group(1) if m else None
        reqs = list(
            dict.fromkeys([*(problem.get("request_ids") or []), *_REQ_RE.findall(combined)])
        )
        errs = list(
            dict.fromkeys(
                [
                    *(problem.get("error_codes") or []),
                    *_ERR_RE.findall(combined),
                    *_HTTP_RE.findall(combined),
                ]
            )
        )
        return acc, ver, reqs, errs

    def _analyze_in_sup(self, view: dict[str, Any]) -> SupervisorAnalysisOutput:
        problem = dict(view.get("problem") or {})
        dialogue = str(view.get("recent_dialogue") or "")
        question = str(problem.get("original_question") or "")
        evidence_catalog = str(view.get("evidence_catalog") or "")
        acc, ver, reqs, errs = self._extract_entities(problem, dialogue)

        needs_account = any(
            kw in question.lower()
            for kw in (
                "403",
                "429",
                "503",
                "上传",
                "失败",
                "排查",
                "req-",
                "账户",
                "配额",
                "日志",
                "冻结",
                "工单",
            )
        ) and not any(kw in question for kw in ("文档", "规范是什么", "需要什么权限"))

        if needs_account and not acc:
            return SupervisorAnalysisOutput(
                account_id=acc,
                product_version=ver,
                error_codes=errs,
                request_ids=reqs,
                summary="问题涉及具体请求失败排查，但当前缺少目标账户 ID，无法查询运行态权限与日志。",
                evidence_sufficient=False,
                information_gaps=["缺少待排查的账户 ID"],
                requires_user_input=True,
                clarification_questions=["请提供需要排查的账户 ID（例如 acc-001）"],
                recommended_action="ask_user",
                reason="缺乏账户 ID 无法精准调用只读业务工具，先向用户追问补充",
            )

        has_evidence = evidence_catalog and evidence_catalog != "暂无已收集证据"
        if has_evidence:
            return SupervisorAnalysisOutput(
                account_id=acc,
                product_version=ver,
                error_codes=errs,
                request_ids=reqs,
                summary="已有调查证据可供判断，进入决策阶段。",
                evidence_sufficient=True,
                recommended_action="answer",
                reason="上下文已包含相关证据，交由 DECIDE 判断下一步",
            )

        return SupervisorAnalysisOutput(
            account_id=acc,
            product_version=ver,
            error_codes=errs,
            request_ids=reqs,
            summary="核心上下文要素已具备，需调派知识检索与运行诊断并行核查。",
            evidence_sufficient=False,
            information_gaps=["待检索版本文档规则与核查业务运行状态"],
            recommended_action="investigate",
            reason="关键参数已齐全，进入 PLAN 制定本轮调查计划",
        )

    def _analyze_in_merge(self, view: dict[str, Any]) -> SupervisorAnalysisOutput:
        problem = dict(view.get("problem") or {})
        round_summaries = list(view.get("round_summaries") or [])
        evidence_catalog = str(view.get("evidence_catalog") or "")
        acc, ver, reqs, errs = self._extract_entities(problem, "")

        has_doc = "[doc:" in evidence_catalog
        has_biz = "[biz:" in evidence_catalog
        gaps: list[str] = []
        conflicts: list[ConflictModel] = []
        abnormal_tasks: list[str] = []

        if round_summaries:
            last_round = round_summaries[-1]
            for t in last_round.get("tasks") or []:
                status = t.get("status")
                if status in ("timeout", "failed", "budget_exhausted"):
                    abnormal_tasks.append(f"{t.get('agent')}({status})")
                    gaps.append(f"{t.get('agent')} 调查未完整结束（状态：{status}）")
                elif status == "no_answer":
                    gaps.append(f"{t.get('agent')} 未在当前数据源中检索到直接依据")

        # 识别可能的证据冲突（例如账户可上传但请求日志又报 E_UPLOAD_403）
        if "can_upload=True" in evidence_catalog and "E_UPLOAD_403" in evidence_catalog:
            eids = _EVIDENCE_ID_RE.findall(evidence_catalog)
            conflicts.append(
                ConflictModel(
                    description="当前账户权限显示允许上传，但历史请求日志仍记录 E_UPLOAD_403，可能存在令牌变更或时间差。",
                    evidence_ids=eids[:4],
                )
            )

        needs_ops = bool(acc)
        if abnormal_tasks:
            sufficient = False
        elif needs_ops:
            sufficient = has_doc and has_biz
            if not has_doc:
                gaps.append("缺少对应版本的产品文档规则依据")
            if not has_biz:
                gaps.append("缺少对应账户或请求的业务运行状态证据")
        else:
            sufficient = has_doc
            if not has_doc:
                gaps.append("知识库未命中匹配的文档说明")

        summary_parts = []
        if has_doc:
            summary_parts.append("已获得产品文档依据")
        if has_biz:
            summary_parts.append("已获得运行态业务证据")
        if abnormal_tasks:
            summary_parts.append("存在部分异常任务: " + ", ".join(abnormal_tasks))

        return SupervisorAnalysisOutput(
            account_id=acc,
            product_version=ver,
            error_codes=errs,
            request_ids=reqs,
            summary="；".join(summary_parts) if summary_parts else "本轮未收集到有效证据",
            evidence_sufficient=sufficient,
            information_gaps=gaps,
            conflicts=conflicts,
            requires_user_input=False,
            recommended_action="answer" if sufficient else "investigate",
            reason="基于跨领域证据汇总与部分失败检查得出阶段结论",
        )

    def _build_plan(self, view: dict[str, Any]) -> InvestigationPlanOutput:
        problem = dict(view.get("problem") or {})
        evidence_catalog = str(view.get("evidence_catalog") or "")
        assessment = view.get("assessment") or {}
        health_notes = " ".join(assessment.get("tool_health_notes") or [])

        question = str(problem.get("original_question") or "")
        account_id = problem.get("account_id")
        ver = problem.get("product_version") or "v2.3"
        reqs = problem.get("request_ids") or []

        has_doc = "[doc:" in evidence_catalog
        has_biz = "[biz:" in evidence_catalog

        tasks: list[PlannedTaskOutput] = []
        if not has_doc:
            tasks.append(
                PlannedTaskOutput(
                    agent="knowledge",
                    objective=f"检索产品 {ver} 文档中与「{question}」相关的权限要求与错误码排查规则",
                    max_tool_attempts=3,
                    max_steps=5,
                    timeout_ms=15000,
                )
            )

        # 若账户已知且缺少业务证据，且并非全部业务工具均已因契约异常禁用，则派发运行诊断任务
        if account_id and not has_biz and "output_validation" not in health_notes:
            req_hint = f"及请求 {reqs[0]} 的日志" if reqs else ""
            tasks.append(
                PlannedTaskOutput(
                    agent="operations",
                    objective=f"查询账户 {account_id} 的权限、配额、服务状态{req_hint}",
                    max_tool_attempts=4,
                    max_steps=6,
                    timeout_ms=15000,
                )
            )

        if not tasks:
            agent_choice: Literal["knowledge", "operations"] = "operations" if account_id else "knowledge"
            tasks.append(
                PlannedTaskOutput(
                    agent=agent_choice,
                    objective=f"针对剩余信息缺口进行补充核查：{question}",
                    max_tool_attempts=3,
                    max_steps=5,
                    timeout_ms=15000,
                )
            )

        return InvestigationPlanOutput(tasks=tasks)

    def _compose_answer(self, view: dict[str, Any]) -> AnswerCompositionOutput:
        problem = dict(view.get("problem") or {})
        evidence_catalog = str(view.get("evidence_catalog") or "")
        assessment = view.get("assessment") or {}
        round_summaries = list(view.get("round_summaries") or [])

        evidence_ids = list(dict.fromkeys(_EVIDENCE_ID_RE.findall(evidence_catalog)))
        lines: list[str] = []

        if "token_scopes=['storage:read']" in evidence_catalog or (
            "can_upload=False" in evidence_catalog and "storage:write" in evidence_catalog
        ):
            lines.append(
                f"排查结论：账户 {problem.get('account_id') or ''} 在 {problem.get('product_version') or 'v2.3'} "
                "上传返回 403，核心原因为当前令牌仅具备 storage:read，缺少 v2.3 强制校验的 storage:write 上传权限。"
            )
            lines.append(
                "建议措施：请为该账户重新生成包含 storage:write 权限范围的访问令牌，并使用相同请求参数复测上传接口。"
            )
        elif "quota_exceeded=True" in evidence_catalog or "E_QUOTA_STORAGE_FULL" in evidence_catalog:
            lines.append("排查结论：账户存储或调用配额已达到上限，触发上传拦截。")
            lines.append("建议措施：请清理历史存储对象或升级账户配额后再重试上传。")
        elif "health=degraded" in evidence_catalog or "INC-" in evidence_catalog:
            lines.append("排查结论：目标区域上传网关当前处于降级状态，导致请求异常。")
            lines.append("建议措施：请暂时切换至健康区域（如 cn-south-1）或待网关事件恢复后重试。")
        elif "status=suspended" in evidence_catalog:
            lines.append("排查结论：账户当前处于 suspended 受限状态，所有上传操作被策略拒绝。")
            lines.append("建议措施：请核对账单或解除账户冻结状态后复测。")
        elif evidence_ids:
            lines.append("排查结论：已结合现有检索与业务检查结果整理依据如下。")
        else:
            lines.append("排查结论：暂未在知识库与当前可用运行状态中获取到足够定位该问题的直接证据。")

        if evidence_ids:
            lines.append("引用依据：" + ", ".join(f"[{eid}]" for eid in evidence_ids))

        round_descs = []
        uncertainties: list[str] = []
        for r in round_summaries:
            t_parts = []
            for t in r.get("tasks") or []:
                t_parts.append(f"{t.get('agent')}:{t.get('status')}(调用{t.get('usage', {}).get('tool_attempts', 0)}次)")
                if not self.omit_uncertainties_in_llm and t.get("status") in (
                    "timeout",
                    "failed",
                    "budget_exhausted",
                ):
                    uncertainties.append(f"任务 {t.get('task_id')} 未完整完成（{t.get('status')}）")
            round_descs.append(f"{r.get('round_id')}[" + ", ".join(t_parts) + "]")

        if not self.omit_uncertainties_in_llm:
            for gap in assessment.get("information_gaps") or []:
                uncertainties.append(str(gap))

        return AnswerCompositionOutput(
            content="\n".join(lines),
            citation_evidence_ids=evidence_ids,
            execution_summary="；".join(round_descs) if round_descs else "未执行额外调查轮次",
            uncertainties=uncertainties,
        )

    def _draft_ticket(self, view: dict[str, Any]) -> DraftTicketOutput:
        problem = dict(view.get("problem") or {})
        evidence_catalog = str(view.get("evidence_catalog") or "")
        assessment = view.get("assessment") or {}
        evidence_ids = list(dict.fromkeys(_EVIDENCE_ID_RE.findall(evidence_catalog)))

        ver = problem.get("product_version") or "v2.3"
        acc = problem.get("account_id") or "未知账户"
        title = f"[{ver}] 账户 {acc} 文件上传异常排查工单"
        description = (
            f"原始问题：{problem.get('original_question', '')}\n"
            f"调查摘要：{assessment.get('summary', '已完成初步排查')}\n"
            f"关联证据：{', '.join(evidence_ids) if evidence_ids else '无'}"
        )
        return DraftTicketOutput(
            title=title,
            description=description,
            evidence_ids=evidence_ids,
            attempted_steps=list(problem.get("attempted_steps") or []),
        )

    def _edit_ticket(self, old_draft: dict[str, Any], edit_request: str) -> DraftTicketOutput:
        old_title = str(old_draft.get("title") or "技术支持工单")
        old_desc = str(old_draft.get("description") or "")
        cleaned_req = edit_request.strip()
        new_title = old_title if "紧急" in old_title or "紧急" not in cleaned_req else f"[紧急] {old_title}"
        new_desc = f"{old_desc}\n补充说明：{cleaned_req}"
        return DraftTicketOutput(
            title=new_title,
            description=new_desc,
            evidence_ids=list(old_draft.get("evidence_ids") or []),
            attempted_steps=list(old_draft.get("attempted_steps") or []),
        )


def create_embedder(
    settings: Settings,
    *,
    use_test_embedder: bool = False,
) -> EmbedderProtocol:
    """根据配置构建向量编码器（测试环境可用确定性哈希向量器，生产/联调默认 BGE-M3）。"""
    if use_test_embedder:
        return DeterministicTestEmbedder(dim=128)
    if settings.embedding_backend == "openai_compatible":
        if not settings.embedding_api_base or not settings.embedding_api_key:
            raise ValueError("openai_compatible embedding 模式必须配置 embedding_api_base 与 embedding_api_key")
        return OpenAICompatibleEmbedder(
            base_url=settings.embedding_api_base,
            api_key=settings.embedding_api_key.get_secret_value(),
            model=settings.embedding_model,
        )
    return SentenceTransformerEmbedder(model_name=settings.embedding_model)


def create_chat_model(
    settings: Settings,
    *,
    use_simulation_llm: bool | None = None,
) -> BaseChatModel:
    """构建对话模型：simulation 模式或未填真实 Key 时使用 SimulationChatModel，否则使用 ChatOpenAI。"""
    if use_simulation_llm is True or (
        use_simulation_llm is None
        and (
            settings.runtime_mode == "simulation"
            or settings.llm_api_key.get_secret_value() == "sk-simulation-placeholder"
        )
    ):
        return SimulationChatModel()

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
    )


def create_simulation_deps(
    *,
    settings: Settings | None = None,
    fault_config: FaultInjectorConfig | None = None,
    verified_account_id: str | None = None,
    llm: BaseChatModel | None = None,
    use_test_embedder: bool = True,
    ticket_store: TicketStoreProtocol | None = None,
    knowledge_retriever: KnowledgeRetrieverProtocol | None = None,
    operations_gateway: OperationsGatewayProtocol | None = None,
) -> RuntimeDeps:
    """组装完全离线可运行的仿真环境 RuntimeDeps（支撑 pytest 与本地前端调试）。"""
    resolved_settings = settings or get_settings()
    resolved_faults = fault_config or FaultInjectorConfig()
    injector = FaultInjector(resolved_faults)
    embedder = create_embedder(resolved_settings, use_test_embedder=use_test_embedder)

    resolved_retriever = knowledge_retriever or InMemoryHybridKnowledgeRetriever(
        embedder=embedder,
        rrf_k=resolved_settings.rrf_k,
        fault_injector=injector,
    )
    resolved_gateway = operations_gateway or SimulationOperationsGateway(
        fault_injector=injector,
    )
    resolved_store = ticket_store or InMemoryTicketStore(fault_injector=injector)
    resolved_llm = llm or create_chat_model(resolved_settings, use_simulation_llm=True)

    return RuntimeDeps.build(
        settings=resolved_settings,
        knowledge_retriever=resolved_retriever,
        operations_gateway=resolved_gateway,
        ticket_store=resolved_store,
        supervisor_llm=resolved_llm,
        agent_llm=resolved_llm,
        verified_account_id=verified_account_id,
        fault_config=resolved_faults,
    )


def create_live_deps(
    *,
    pool: AsyncConnectionPool[Any],
    settings: Settings | None = None,
    fault_config: FaultInjectorConfig | None = None,
    verified_account_id: str | None = None,
    use_simulation_llm: bool = False,
    use_test_embedder: bool = False,
) -> RuntimeDeps:
    """组装连接真实 PostgreSQL（pgvector + tickets + Checkpointer）的 RuntimeDeps。"""
    resolved_settings = settings or get_settings()
    resolved_faults = fault_config or FaultInjectorConfig()
    injector = FaultInjector(resolved_faults)
    embedder = create_embedder(resolved_settings, use_test_embedder=use_test_embedder)

    retriever = PgVectorHybridKnowledgeRetriever(
        pool=pool,
        embedder=embedder,
        rrf_k=resolved_settings.rrf_k,
        fault_injector=injector,
    )
    gateway = SimulationOperationsGateway(fault_injector=injector)
    store = PostgresTicketStore(pool=pool, fault_injector=injector)
    llm = create_chat_model(resolved_settings, use_simulation_llm=use_simulation_llm)

    return RuntimeDeps.build(
        settings=resolved_settings,
        knowledge_retriever=retriever,
        operations_gateway=gateway,
        ticket_store=store,
        supervisor_llm=llm,
        agent_llm=llm,
        verified_account_id=verified_account_id,
        fault_config=resolved_faults,
    )
