from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from supportflow.context_policy import format_tool_observation
from supportflow.deps import RuntimeDeps
from supportflow.schemas import (
    TOOL_INPUT_SCHEMAS,
    TOOL_OUTPUT_SCHEMAS,
    AgentFinalSummaryOutput,
    DefiniteToolError,
    GetAccountPermissionsOutput,
    GetAccountQuotaOutput,
    GetRequestLogsOutput,
    GetServiceStatusOutput,
    ReadDocumentChunkArgs,
    ReadDocumentChunkOutput,
    SearchDocumentsArgs,
    SearchDocumentsOutput,
    TransientToolError,
)
from supportflow.state import (
    AgentLocalState,
    DiagnosisResult,
    Evidence,
    ExecutionError,
    LogicalCallRecord,
    make_logical_call_id,
)

KNOWLEDGE_TOOL_NAMES = ("search_documents", "read_document_chunk")
OPERATIONS_TOOL_NAMES = (
    "get_account_permissions",
    "get_account_quota",
    "get_service_status",
    "get_request_logs",
)

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "search_documents": "按关键词/语义和产品版本检索官方产品文档与故障排查指南（混合检索 + RRF）。",
    "read_document_chunk": "精确读取指定文档编号、版本与分块编号的完整原文内容。",
    "get_account_permissions": "只读查询指定账户的当前状态、权限清单、令牌作用域及是否允许上传。",
    "get_account_quota": "只读查询指定账户的存储配额使用量与每日 API 调用量。",
    "get_service_status": "只读查询上传、认证、存储或网关服务的当前区域健康状态与事件编号。",
    "get_request_logs": "只读查询指定账户下某请求 ID 或时间段内的服务端请求日志与错误码。",
}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _build_openai_tool_specs(tool_names: tuple[str, ...]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name in tool_names:
        schema_cls = TOOL_INPUT_SCHEMAS[name]
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _TOOL_DESCRIPTIONS[name],
                    "parameters": schema_cls.model_json_schema(),
                },
            }
        )
    return specs


def _parse_final_summary(content: str) -> AgentFinalSummaryOutput | None:
    text = content.strip()
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    candidate = match.group(1) if match else text
    if not candidate.startswith("{"):
        return None
    try:
        return AgentFinalSummaryOutput.model_validate_json(candidate)
    except ValidationError:
        return None


def _classify_business_result(
    tool_name: str,
    parsed: BaseModel,
) -> tuple[Literal["normal", "business_error", "no_data"], str]:
    if isinstance(parsed, GetAccountPermissionsOutput):
        if parsed.status != "active" or not parsed.can_upload:
            return (
                "business_error",
                (
                    f"账户 {parsed.account_id} 状态={parsed.status}, "
                    f"can_upload={parsed.can_upload}, "
                    f"permissions={parsed.permissions}, token_scopes={parsed.token_scopes}"
                ),
            )
        return (
            "normal",
            (
                f"账户 {parsed.account_id} 正常且具备上传能力 "
                f"(token_scopes={parsed.token_scopes})"
            ),
        )

    if isinstance(parsed, GetAccountQuotaOutput):
        if parsed.quota_exceeded:
            return (
                "business_error",
                (
                    f"账户 {parsed.account_id} 配额已超限: "
                    f"storage={parsed.storage_used_bytes}/{parsed.storage_limit_bytes} bytes, "
                    f"api={parsed.daily_api_calls}/{parsed.daily_api_limit}"
                ),
            )
        return (
            "normal",
            (
                f"账户 {parsed.account_id} 配额正常: "
                f"storage={parsed.storage_used_bytes}/{parsed.storage_limit_bytes} bytes, "
                f"api={parsed.daily_api_calls}/{parsed.daily_api_limit}"
            ),
        )

    if isinstance(parsed, GetServiceStatusOutput):
        if parsed.health != "healthy":
            return (
                "business_error",
                (
                    f"服务 {parsed.service_name} ({parsed.region}) 状态={parsed.health}, "
                    f"incident={parsed.incident_id}: {parsed.summary}"
                ),
            )
        return (
            "normal",
            f"服务 {parsed.service_name} ({parsed.region}) 健康: {parsed.summary}",
        )

    if isinstance(parsed, GetRequestLogsOutput):
        if not parsed.entries:
            return (
                "no_data",
                f"账户 {parsed.account_id} 在指定过滤条件下未查询到匹配请求日志",
            )
        has_err = any(e.http_status >= 400 or e.error_code for e in parsed.entries)
        summaries = [
            f"{e.request_id}({e.endpoint} -> HTTP {e.http_status}, code={e.error_code}, reason={e.reason})"
            for e in parsed.entries[:5]
        ]
        return (
            "business_error" if has_err else "normal",
            f"账户 {parsed.account_id} 命中 {len(parsed.entries)} 条请求日志: "
            + "; ".join(summaries),
        )

    return ("normal", parsed.model_dump_json())


class SubAgentRunner:
    """通用可复用子 Agent 循环执行器。

    支持 knowledge 与 operations 两类独立子 Agent 的统一循环执行。
    """

    def __init__(self, deps: RuntimeDeps) -> None:
        self._deps = deps

    def _allowed_tool_names(self, agent_name: str) -> tuple[str, ...]:
        if agent_name == "knowledge":
            return KNOWLEDGE_TOOL_NAMES
        return OPERATIONS_TOOL_NAMES

    def _max_retries_for_tool(self, agent_name: str, tool_name: str) -> int:
        # 知识诊断禁止自动重试；运行诊断同一次逻辑调用最多重试 1 次
        if agent_name == "operations":
            return 1
        return 0

    def _remaining_time_s(self, local: AgentLocalState) -> float:
        return local["deadline_monotonic"] - self._deps.clock.monotonic()

    def _record_error(
        self,
        local: AgentLocalState,
        *,
        kind: ExecutionError["kind"],
        message: str,
        tool_name: str | None,
        logical_call_id: str | None,
    ) -> ExecutionError:
        err: ExecutionError = {
            "kind": kind,
            "message": message,
            "tool_name": tool_name,
            "logical_call_id": logical_call_id,
        }
        local["errors"].append(err)
        return err

    async def _invoke_knowledge_tool(
        self,
        tool_name: str,
        validated_model: BaseModel,
    ) -> dict[str, Any]:
        retriever = self._deps.knowledge_retriever
        if tool_name == "search_documents":
            assert isinstance(validated_model, SearchDocumentsArgs)
            return await retriever.search_documents(validated_model)
        if tool_name == "read_document_chunk":
            assert isinstance(validated_model, ReadDocumentChunkArgs)
            return await retriever.read_document_chunk(validated_model)
        raise DefiniteToolError(f"未知知识工具: {tool_name}")

    async def run_task(self, local: AgentLocalState) -> DiagnosisResult:
        task = local["task"]
        allowed_names = self._allowed_tool_names(task["agent"])

        while True:
            if local["usage"]["agent_steps"] >= task["max_steps"]:
                return self._build_result(
                    local,
                    status="budget_exhausted",
                    summary=self._fallback_summary(local, reason="达到单任务最大思考步数上限"),
                )
            if local["usage"]["tool_attempts"] >= task["max_tool_attempts"]:
                return self._build_result(
                    local,
                    status="budget_exhausted",
                    summary=self._fallback_summary(local, reason="达到单任务最大工具调用次数上限"),
                )
            if self._remaining_time_s(local) <= 0:
                return self._build_result(
                    local,
                    status="timeout",
                    summary=self._fallback_summary(local, reason="达到任务剩余时间上限"),
                )

            active_names = tuple(
                name for name in allowed_names if name not in local["disabled_tools"]
            )
            if not active_names:
                return self._build_result(
                    local,
                    status="failed",
                    summary=self._fallback_summary(
                        local,
                        reason="当前可用工具均因返回契约损坏被局部熔断",
                    ),
                )

            local["usage"]["agent_steps"] += 1
            evidence_count_before = len(local["evidence"])

            tool_specs = _build_openai_tool_specs(active_names)
            bound_llm = self._deps.agent_llm.bind_tools(tool_specs)
            ai_msg = await bound_llm.ainvoke(local["messages"])
            if not isinstance(ai_msg, AIMessage):
                ai_msg = AIMessage(content=str(ai_msg))
            local["messages"].append(ai_msg)

            # 1) 处理底层 JSON 解析失败的无效工具调用，按 argument_validation 回喂模型自纠
            for bad_call in getattr(ai_msg, "invalid_tool_calls", []) or []:
                bad_name = bad_call.get("name") or "unknown"
                bad_id = bad_call.get("id") or f"invalid-{uuid.uuid4().hex[:8]}"
                bad_err = bad_call.get("error") or "JSON 参数格式无效"
                self._record_error(
                    local,
                    kind="argument_validation",
                    message=f"工具 {bad_name} 参数解析失败: {bad_err}",
                    tool_name=bad_name,
                    logical_call_id=None,
                )
                local["messages"].append(
                    ToolMessage(
                        tool_call_id=bad_id,
                        content=f"参数格式解析失败（argument_validation）：{bad_err}，请修正后重试。",
                    )
                )

            tool_calls = list(getattr(ai_msg, "tool_calls", []) or [])
            invalid_calls = list(getattr(ai_msg, "invalid_tool_calls", []) or [])

            # 2) 模型主动结束工具调用，给出阶段结论
            if not tool_calls and not invalid_calls:
                return self._finalize_from_ai_message(local, str(ai_msg.content))

            # 3) 逐个执行本步生成的工具调用
            budget_hit_reason: str | None = None
            for call in tool_calls:
                tool_name = str(call["name"])
                call_id = str(call.get("id") or f"tc-{uuid.uuid4().hex[:8]}")
                raw_args = call.get("args") or {}

                if tool_name in local["disabled_tools"]:
                    local["messages"].append(
                        ToolMessage(
                            tool_call_id=call_id,
                            content=(
                                f"工具 {tool_name} 因返回契约异常已在当前轮次熔断停用"
                                f"（原因：{local['disabled_tools'][tool_name]}），请勿继续调用。"
                            ),
                        )
                    )
                    continue

                if tool_name not in allowed_names:
                    self._record_error(
                        local,
                        kind="argument_validation",
                        message=f"工具 {tool_name} 不在当前 Agent 允许范围 {allowed_names} 内",
                        tool_name=tool_name,
                        logical_call_id=None,
                    )
                    local["messages"].append(
                        ToolMessage(
                            tool_call_id=call_id,
                            content=f"参数错误（argument_validation）：无权或不存在工具 {tool_name}。",
                        )
                    )
                    continue

                # KP / OP：Pydantic 严格入参校验（失败不扣减 tool_attempts，直接反馈给模型自纠）
                input_schema = TOOL_INPUT_SCHEMAS[tool_name]
                try:
                    validated_model = input_schema.model_validate(raw_args)
                except ValidationError as exc:
                    err_text = exc.errors(include_url=False)
                    self._record_error(
                        local,
                        kind="argument_validation",
                        message=f"{tool_name} 入参校验失败: {err_text}",
                        tool_name=tool_name,
                        logical_call_id=None,
                    )
                    local["messages"].append(
                        ToolMessage(
                            tool_call_id=call_id,
                            content=(
                                f"参数校验失败（argument_validation），请修正以下字段后重新调用：\n"
                                f"{json.dumps(err_text, ensure_ascii=False)}"
                            ),
                        )
                    )
                    continue

                validated_dict = validated_model.model_dump()

                # OP 租户沙箱校验（对应第 37 点）：校验 account_id 必须等于会话已核验账户
                if (
                    self._deps.verified_account_id is not None
                    and "account_id" in validated_dict
                    and validated_dict["account_id"] != self._deps.verified_account_id
                ):
                    msg = (
                        f"租户沙箱拦截（argument_validation）：禁止查询非核验账户 "
                        f"{validated_dict['account_id']}，请使用当前会话合法账户 "
                        f"{self._deps.verified_account_id}。"
                    )
                    self._record_error(
                        local,
                        kind="argument_validation",
                        message=msg,
                        tool_name=tool_name,
                        logical_call_id=None,
                    )
                    local["messages"].append(ToolMessage(tool_call_id=call_id, content=msg))
                    continue

                # RECORD：按 task_id + tool_name + 规范化参数 生成跨循环持久的逻辑调用记录
                logical_call_id = make_logical_call_id(
                    task["task_id"], tool_name, validated_dict
                )
                record: LogicalCallRecord | None = local["logical_calls"].get(logical_call_id)
                if record is None:
                    record = {
                        "logical_call_id": logical_call_id,
                        "tool_name": tool_name,
                        "args": dict(validated_dict),
                        "attempts_started": 0,
                        "retry_count": 0,
                        "retry_authorized": False,
                        "last_error": None,
                    }
                    local["logical_calls"][logical_call_id] = record

                max_retries = self._max_retries_for_tool(task["agent"], tool_name)
                raw_result: dict[str, Any] | None = None

                # BUDGET -> OT -> RETRY -> MARK 内部闭环（重试直接复用原参数，不重走 LLM）
                while True:
                    if local["usage"]["tool_attempts"] >= task["max_tool_attempts"]:
                        budget_hit_reason = "达到单任务最大工具调用次数上限"
                        break
                    remaining_s = self._remaining_time_s(local)
                    if remaining_s <= 0:
                        budget_hit_reason = "达到任务时间预算上限"
                        break

                    first_attempt = record["attempts_started"] == 0
                    approved_retry = (
                        record["attempts_started"] == 1
                        and record["retry_authorized"]
                        and record["retry_count"] <= max_retries
                    )
                    if not (first_attempt or approved_retry):
                        limit_msg = (
                            f"逻辑调用 {logical_call_id}（{tool_name}）已达到尝试/重试上限"
                            f"（attempts={record['attempts_started']}, retries={record['retry_count']}），"
                            "禁止跨循环重复相同参数调用，请切换工具或基于现有证据总结。"
                        )
                        self._record_error(
                            local,
                            kind="tool_error",
                            message=limit_msg,
                            tool_name=tool_name,
                            logical_call_id=logical_call_id,
                        )
                        local["messages"].append(
                            ToolMessage(tool_call_id=call_id, content=limit_msg)
                        )
                        break

                    # OT：实际发起调用前先记录已发起尝试并计入工具调用预算
                    record["attempts_started"] += 1
                    record["retry_authorized"] = False
                    local["usage"]["tool_attempts"] += 1

                    call_timeout_s = min(
                        self._deps.settings.default_tool_timeout_ms / 1000.0,
                        remaining_s,
                    )
                    try:
                        if tool_name in KNOWLEDGE_TOOL_NAMES:
                            raw_result = await self._invoke_knowledge_tool(
                                tool_name, validated_model
                            )
                        else:
                            raw_result = await self._deps.operations_gateway.invoke_tool(
                                tool_name,
                                dict(record["args"]),
                                call_timeout_s,
                            )
                        break
                    except TransientToolError as exc:
                        err = self._record_error(
                            local,
                            kind=exc.kind,
                            message=str(exc),
                            tool_name=tool_name,
                            logical_call_id=logical_call_id,
                        )
                        record["last_error"] = err

                        can_retry = (
                            max_retries >= 1
                            and record["attempts_started"] == 1
                            and record["retry_count"] == 0
                            and local["usage"]["tool_attempts"] < task["max_tool_attempts"]
                            and self._remaining_time_s(local) > 0
                        )
                        if can_retry:
                            # MARK：授权复用原参数直接重试一次，回到 BUDGET
                            record["retry_count"] = 1
                            record["retry_authorized"] = True
                            continue

                        local["messages"].append(
                            ToolMessage(
                                tool_call_id=call_id,
                                content=(
                                    f"工具 {tool_name} 瞬时调用失败（{exc.kind}），"
                                    f"无法继续重试：{exc}。请注意区分调用失败与业务异常。"
                                ),
                            )
                        )
                        break
                    except DefiniteToolError as exc:
                        err = self._record_error(
                            local,
                            kind="tool_error",
                            message=str(exc),
                            tool_name=tool_name,
                            logical_call_id=logical_call_id,
                        )
                        record["last_error"] = err
                        local["messages"].append(
                            ToolMessage(
                                tool_call_id=call_id,
                                content=f"工具 {tool_name} 确定性调用失败：{exc}",
                            )
                        )
                        break
                    except TimeoutError:
                        # 若已到外层任务硬截止时间，向上抛出交由 WRAP 统一转为 timeout 结果
                        if self._remaining_time_s(local) <= 0:
                            raise
                        raise

                if budget_hit_reason is not None:
                    break
                if raw_result is None:
                    continue

                # KV / OV：Pydantic 返回结构校验，失败按 output_validation 熔断该工具，不让模型瞎改参数重试
                output_schema = TOOL_OUTPUT_SCHEMAS[tool_name]
                try:
                    parsed_output = output_schema.model_validate(raw_result)
                except ValidationError as exc:
                    err_detail = f"工具 {tool_name} 服务端返回结构契约损坏: {exc.errors(include_url=False)}"
                    self._record_error(
                        local,
                        kind="output_validation",
                        message=err_detail,
                        tool_name=tool_name,
                        logical_call_id=logical_call_id,
                    )
                    local["disabled_tools"][tool_name] = "返回契约校验失败（output_validation）"
                    local["messages"].append(
                        ToolMessage(
                            tool_call_id=call_id,
                            content=(
                                f"服务端返回契约损坏（output_validation），本轮已自动停用 {tool_name}。"
                                "禁止继续重试或修改参数调用该工具，请改用其他工具或基于现有证据总结。"
                            ),
                        )
                    )
                    continue

                # KD / OD：落盘结构化 Evidence 并生成带截断与定界标签的 ToolMessage
                added_ids = self._record_evidence(
                    local=local,
                    tool_name=tool_name,
                    logical_call_id=logical_call_id,
                    query_params=validated_dict,
                    parsed_output=parsed_output,
                )
                obs_text = format_tool_observation(
                    tool_name,
                    parsed_output.model_dump(),
                    max_chars=self._deps.settings.max_tool_output_chars,
                    evidence_ids=added_ids,
                )
                local["messages"].append(ToolMessage(tool_call_id=call_id, content=obs_text))

            if budget_hit_reason is not None:
                status: Literal["timeout", "budget_exhausted"] = (
                    "timeout" if self._remaining_time_s(local) <= 0 else "budget_exhausted"
                )
                return self._build_result(
                    local,
                    status=status,
                    summary=self._fallback_summary(local, reason=budget_hit_reason),
                )

            # 25 行纯规则闭环保护：连续 2 步无任何新证据产出则提前收敛退出，防止无意义空转
            if len(local["evidence"]) > evidence_count_before:
                local["consecutive_no_evidence_steps"] = 0
            else:
                local["consecutive_no_evidence_steps"] += 1

            if local["consecutive_no_evidence_steps"] >= 2:
                exit_status: Literal["completed", "no_answer", "failed"]
                if local["evidence"]:
                    exit_status = "completed"
                elif local["disabled_tools"] or any(
                    e["kind"] in ("timeout", "connection", "output_validation", "tool_error")
                    for e in local["errors"]
                ):
                    exit_status = "failed"
                else:
                    exit_status = "no_answer"

                return self._build_result(
                    local,
                    status=exit_status,
                    summary=self._fallback_summary(
                        local,
                        reason="连续两步调查未产出新证据，触发防空转保护提前收敛",
                    ),
                )

    def _record_evidence(
        self,
        *,
        local: AgentLocalState,
        tool_name: str,
        logical_call_id: str,
        query_params: dict[str, Any],
        parsed_output: BaseModel,
    ) -> list[str]:
        added_ids: list[str] = []

        if isinstance(parsed_output, SearchDocumentsOutput):
            for chunk in parsed_output.chunks:
                eid = f"doc:{chunk.document_id}:{chunk.document_version}:{chunk.chunk_id}"
                local["evidence"][eid] = {
                    "evidence_id": eid,
                    "content": f"[{chunk.title}] {chunk.text}",
                    "source": {
                        "kind": "document",
                        "document_id": chunk.document_id,
                        "document_version": chunk.document_version,
                        "chunk_id": chunk.chunk_id,
                    },
                }
                added_ids.append(eid)
            return added_ids

        if isinstance(parsed_output, ReadDocumentChunkOutput):
            if parsed_output.found and parsed_output.chunk is not None:
                c = parsed_output.chunk
                eid = f"doc:{c.document_id}:{c.document_version}:{c.chunk_id}"
                local["evidence"][eid] = {
                    "evidence_id": eid,
                    "content": f"[{c.title}] {c.text}",
                    "source": {
                        "kind": "document",
                        "document_id": c.document_id,
                        "document_version": c.document_version,
                        "chunk_id": c.chunk_id,
                    },
                }
                added_ids.append(eid)
            return added_ids

        # 运行诊断业务证据
        outcome, content_summary = _classify_business_result(tool_name, parsed_output)
        queried_at = getattr(parsed_output, "checked_at", self._deps.clock.now_iso())
        seq = len(local["evidence"]) + 1
        eid = f"biz:{tool_name}:{logical_call_id[-6:]}:{seq}"
        local["evidence"][eid] = {
            "evidence_id": eid,
            "content": content_summary,
            "source": {
                "kind": "business",
                "tool_name": tool_name,
                "logical_call_id": logical_call_id,
                "query_params": dict(query_params),
                "queried_at": queried_at,
                "outcome": outcome,
            },
        }
        added_ids.append(eid)
        return added_ids

    def _finalize_from_ai_message(
        self,
        local: AgentLocalState,
        raw_content: str,
    ) -> DiagnosisResult:
        parsed = _parse_final_summary(raw_content)
        if parsed is not None:
            status: Literal["completed", "no_answer"] = parsed.status
            summary = parsed.summary
        else:
            cleaned = raw_content.strip()
            status = "completed" if local["evidence"] else "no_answer"
            summary = cleaned or self._fallback_summary(local, reason="Agent 完成调查")

        return self._build_result(local, status=status, summary=summary)

    def _fallback_summary(self, local: AgentLocalState, *, reason: str) -> str:
        ev_count = len(local["evidence"])
        err_count = len(local["errors"])
        if ev_count > 0:
            snippets = [item["content"][:140] for item in list(local["evidence"].values())[:3]]
            return f"{reason}。已收集 {ev_count} 条证据（错误 {err_count} 条）：" + " | ".join(snippets)
        return f"{reason}，未收集到有效证据（记录错误 {err_count} 条）。"

    def _build_result(
        self,
        local: AgentLocalState,
        *,
        status: DiagnosisResult["status"],
        summary: str,
    ) -> DiagnosisResult:
        return {
            "round_id": local["round_id"],
            "task_id": local["task"]["task_id"],
            "agent": local["task"]["agent"],
            "status": status,
            "summary": summary,
            "evidence": list(local["evidence"].values()),
            "errors": list(local["errors"]),
            "usage": dict(local["usage"]),
        }
