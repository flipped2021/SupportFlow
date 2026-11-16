from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from supportflow.context_policy import project_state_for_supervisor, wrap_untrusted_data
from supportflow.deps import RuntimeDeps
from supportflow.schemas import (
    AnswerCompositionOutput,
    ClarificationResponse,
    InvestigationPlanOutput,
    SupervisorAnalysisOutput,
)
from supportflow.state import (
    Answer,
    Assessment,
    Decision,
    InvestigationRound,
    ProblemContext,
    RoundPlan,
    SupportState,
    TaskPlan,
)

_TICKET_INTENT_KEYWORDS = ("工单", "提单", "创建工单", "转人工", "ticket", "submit ticket")


def compute_budget_status(state: SupportState) -> dict[str, Any]:
    """严格按约定口径核算全局已消耗预算与剩余预算。

    - 已用轮次 = len(state["rounds"])
    - 已用工具调用次数 = 所有已完成轮次中各任务 usage["tool_attempts"] 之和
    - 已用调查耗时 = 所有已完成轮次 elapsed_ms 之和
    """
    budget = state["budget"]
    rounds_used = len(state["rounds"])
    tool_attempts_used = sum(
        res["usage"]["tool_attempts"]
        for round_data in state["rounds"].values()
        for res in (round_data["results"] or [])
    )
    investigation_ms_used = sum(
        (round_data["elapsed_ms"] or 0)
        for round_data in state["rounds"].values()
        if round_data["results"] is not None
    )

    remaining_rounds = max(0, budget["max_rounds"] - rounds_used)
    remaining_tool_attempts = max(0, budget["max_tool_attempts"] - tool_attempts_used)
    remaining_investigation_ms = max(0, budget["max_investigation_ms"] - investigation_ms_used)

    exhausted = (
        remaining_rounds <= 0
        or remaining_tool_attempts <= 0
        or remaining_investigation_ms <= 0
    )
    return {
        "rounds_used": rounds_used,
        "tool_attempts_used": tool_attempts_used,
        "investigation_ms_used": investigation_ms_used,
        "remaining_rounds": remaining_rounds,
        "remaining_tool_attempts": remaining_tool_attempts,
        "remaining_investigation_ms": remaining_investigation_ms,
        "exhausted": exhausted,
    }


def _user_wants_ticket(state: SupportState) -> bool:
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            text = str(msg.content).lower()
            if any(kw in text for kw in _TICKET_INTENT_KEYWORDS):
                return True
    return False


def _merge_problem_context(
    old: ProblemContext,
    analysis: SupervisorAnalysisOutput,
    *,
    verified_account_id: str | None,
) -> ProblemContext:
    merged_error_codes = list(
        dict.fromkeys([*old["error_codes"], *[c for c in analysis.error_codes if c]])
    )
    merged_request_ids = list(
        dict.fromkeys([*old["request_ids"], *[r for r in analysis.request_ids if r]])
    )
    merged_facts = {**old["known_facts"], **analysis.known_facts_updates}

    resolved_account = verified_account_id or old["account_id"] or analysis.account_id
    resolved_version = old["product_version"] or analysis.product_version

    return {
        "original_question": old["original_question"],
        "account_id": resolved_account,
        "product_version": resolved_version,
        "error_codes": merged_error_codes,
        "request_ids": merged_request_ids,
        "known_facts": merged_facts,
        "attempted_steps": list(old["attempted_steps"]),
    }


def _collect_tool_health_notes(state: SupportState) -> list[str]:
    notes: list[str] = []
    for round_id, rdata in state["rounds"].items():
        for res in rdata["results"] or []:
            for err in res["errors"]:
                if err["kind"] == "output_validation" and err["tool_name"]:
                    notes.append(
                        f"[{round_id}] 工具 {err['tool_name']} 返回契约损坏并在当轮熔断：{err['message']}"
                    )
                elif err["kind"] in ("timeout", "connection", "tool_error") and err["tool_name"]:
                    notes.append(
                        f"[{round_id}] 工具 {err['tool_name']} 发生 {err['kind']}：{err['message']}"
                    )
    return list(dict.fromkeys(notes))


def _apply_budget_to_tasks(
    planned: InvestigationPlanOutput,
    *,
    round_id: str,
    remaining_tool_attempts: int,
    remaining_investigation_ms: int,
    max_agent_steps_per_task: int,
    default_round_timeout_ms: int,
    default_task_timeout_ms: int,
) -> RoundPlan:
    """强制约束本轮各任务预算：保证 max_tool_attempts 之和不超过全局剩余工具额度。"""
    round_timeout_ms = max(
        500,
        min(default_round_timeout_ms, remaining_investigation_ms),
    )

    candidate_tasks = list(planned.tasks)
    if remaining_tool_attempts <= 0:
        candidate_tasks = []
    elif len(candidate_tasks) > remaining_tool_attempts:
        # 若全局剩余次数不足 2 次，保留优先级最高的前 N 个任务，确保每个派发任务至少拥有 1 次调用额度
        candidate_tasks = candidate_tasks[:remaining_tool_attempts]

    allocated_tasks: list[TaskPlan] = []
    remaining_pool = remaining_tool_attempts
    num_tasks = len(candidate_tasks)

    for idx, t in enumerate(candidate_tasks):
        tasks_left_after = num_tasks - idx - 1
        max_for_this = max(1, remaining_pool - tasks_left_after)
        assigned_attempts = min(max(1, t.max_tool_attempts), max_for_this)
        remaining_pool -= assigned_attempts

        assigned_steps = min(max(1, t.max_steps), max_agent_steps_per_task)
        assigned_timeout = min(max(500, t.timeout_ms), default_task_timeout_ms, round_timeout_ms)

        allocated_tasks.append(
            {
                "task_id": f"{round_id}:{t.agent}",
                "agent": t.agent,
                "objective": t.objective,
                "max_tool_attempts": assigned_attempts,
                "max_steps": assigned_steps,
                "timeout_ms": assigned_timeout,
            }
        )

    return {
        "round_id": round_id,
        "tasks": allocated_tasks,
        "timeout_ms": round_timeout_ms,
    }


async def sup_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """SUP 节点：分析问题上下文、提取实体并识别是否缺信息/需调查/已有充分证据。"""
    deps = runtime.context
    projected = project_state_for_supervisor(
        state,
        max_history_messages=deps.settings.max_main_history_messages,
    )
    budget_status = compute_budget_status(state)

    structured_llm = deps.supervisor_llm.with_structured_output(SupervisorAnalysisOutput)
    prompt = [
        SystemMessage(
            content=(
                "你是 SupportFlow 的 Supervisor。请基于问题上下文与已有证据分析：\n"
                "1. 提取产品版本、账户 ID、错误码与请求 ID。\n"
                "2. 若缺少关键定位信息（例如要排查哪个账户），将 recommended_action 设为 'ask_user' 并列出追问问题。\n"
                "3. 若仍需查文档或业务运行状态且预算充足，设 recommended_action 为 'investigate'。\n"
                "4. 若现有证据已足够回答问题或预算耗尽，设 recommended_action 为 'answer' 或 'draft_ticket'（仅当用户明确要求创建工单时）。"
            )
        ),
        HumanMessage(
            content=(
                f"当前问题与证据投影：\n{wrap_untrusted_data('supervisor_view', json.dumps(projected, ensure_ascii=False, indent=2))}\n"
                f"预算状态：{json.dumps(budget_status, ensure_ascii=False)}"
            )
        ),
    ]
    analysis = await structured_llm.ainvoke(prompt)
    assert isinstance(analysis, SupervisorAnalysisOutput)

    updated_problem = _merge_problem_context(
        state["problem"],
        analysis,
        verified_account_id=deps.verified_account_id,
    )

    assessment: Assessment = {
        "summary": analysis.summary,
        "evidence_sufficient": analysis.evidence_sufficient,
        "information_gaps": list(analysis.information_gaps),
        "conflicts": [
            {"description": c.description, "evidence_ids": list(c.evidence_ids)}
            for c in analysis.conflicts
        ],
        "tool_health_notes": _collect_tool_health_notes(state),
    }

    clarification_questions = list(analysis.clarification_questions)
    # 若已有核验账户，避免不必要追问账户 ID
    if updated_problem["account_id"] and clarification_questions:
        clarification_questions = [
            q for q in clarification_questions if "账户" not in q or len(clarification_questions) > 1
        ]

    action = analysis.recommended_action
    if action == "ask_user" and not clarification_questions:
        action = "investigate"
    if action == "investigate" and budget_status["exhausted"]:
        action = "answer"

    decision: Decision = {
        "action": action,
        "reason": analysis.reason,
    }
    return {
        "problem": updated_problem,
        "assessment": assessment,
        "decision": decision,
        "clarification_questions": clarification_questions if action == "ask_user" else [],
    }


async def ask_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """ASK 节点：触发 interrupt 暂停等待用户补充必要信息，恢复后并入 ProblemContext。"""
    questions = state["clarification_questions"] or ["请补充需要排查的账户 ID 或更多错误现象"]
    raw_response = interrupt(
        {
            "type": "clarification",
            "case_id": state["case_id"],
            "questions": questions,
        }
    )

    if isinstance(raw_response, str):
        validated = ClarificationResponse(text=raw_response)
    else:
        validated = ClarificationResponse.model_validate(raw_response)

    old_problem = state["problem"]
    # 尝试从自由文本中辅助提取 acc-xxx 或 req-xxx（若结构化字段未单独填）
    inferred_account = validated.account_id
    if inferred_account is None:
        acc_match = re.search(r"\b(acc-[A-Za-z0-9_-]+)\b", validated.text)
        if acc_match:
            inferred_account = acc_match.group(1)

    inferred_requests = list(validated.request_ids)
    if not inferred_requests:
        inferred_requests = re.findall(r"\b(req-[A-Za-z0-9_-]+)\b", validated.text)

    inferred_version = validated.product_version
    if inferred_version is None:
        ver_match = re.search(r"\b(v\d+\.\d+(?:\.\d+)?)\b", validated.text)
        if ver_match:
            inferred_version = ver_match.group(1)

    resolved_account = (
        runtime.context.verified_account_id
        or old_problem["account_id"]
        or inferred_account
    )
    updated_problem: ProblemContext = {
        "original_question": old_problem["original_question"],
        "account_id": resolved_account,
        "product_version": old_problem["product_version"] or inferred_version,
        "error_codes": list(old_problem["error_codes"]),
        "request_ids": list(dict.fromkeys([*old_problem["request_ids"], *inferred_requests])),
        "known_facts": {**old_problem["known_facts"], **validated.extra_facts},
        "attempted_steps": list(old_problem["attempted_steps"]),
    }

    return {
        "problem": updated_problem,
        "messages": [HumanMessage(content=validated.text)],
        "clarification_questions": [],
    }


async def plan_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """PLAN 节点：生成本轮调查计划与稳定 round_id/task_id，并在执行前落盘至 State。"""
    deps = runtime.context
    budget_status = compute_budget_status(state)
    round_num = len(state["rounds"]) + 1
    round_id = f"{state['case_id']}:r{round_num}"

    projected = project_state_for_supervisor(
        state,
        max_history_messages=deps.settings.max_main_history_messages,
    )
    structured_llm = deps.supervisor_llm.with_structured_output(InvestigationPlanOutput)
    prompt = [
        SystemMessage(
            content=(
                "你是 SupportFlow 的调查规划器。请规划本轮调查任务（1~2 个任务，每个 agent 每轮最多出现 1 次）：\n"
                "- knowledge: 检索产品文档、权限要求与错误码规范。\n"
                "- operations: 查询账户权限、配额、服务状态与请求日志。\n"
                f"全局剩余工具调用额度为 {budget_status['remaining_tool_attempts']} 次，"
                f"剩余调查时间为 {budget_status['remaining_investigation_ms']}ms。"
            )
        ),
        HumanMessage(
            content=wrap_untrusted_data(
                "plan_context",
                json.dumps(projected, ensure_ascii=False, indent=2),
            )
        ),
    ]
    planned = await structured_llm.ainvoke(prompt)
    assert isinstance(planned, InvestigationPlanOutput)

    plan = _apply_budget_to_tasks(
        planned,
        round_id=round_id,
        remaining_tool_attempts=budget_status["remaining_tool_attempts"],
        remaining_investigation_ms=budget_status["remaining_investigation_ms"],
        max_agent_steps_per_task=state["budget"]["max_agent_steps_per_task"],
        default_round_timeout_ms=deps.settings.default_round_timeout_ms,
        default_task_timeout_ms=deps.settings.default_task_timeout_ms,
    )

    new_round: InvestigationRound = {
        "plan": plan,
        "results": None,
        "elapsed_ms": None,
    }
    return {
        "current_round_id": round_id,
        "rounds": {
            **state["rounds"],
            round_id: new_round,
        },
    }


async def merge_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """MERGE 节点：汇总本轮结果，合并 Evidence 并识别部分失败、证据冲突与信息缺口。"""
    deps = runtime.context
    round_id = state["current_round_id"]
    assert round_id is not None and round_id in state["rounds"]

    current_results = state["rounds"][round_id]["results"] or []
    merged_evidence = dict(state["evidence"])
    attempted_steps = list(state["problem"]["attempted_steps"])

    for res in current_results:
        for ev in res["evidence"]:
            merged_evidence[ev["evidence_id"]] = ev
            src = ev["source"]
            if src["kind"] == "business":
                step_desc = f"调用 {src['tool_name']}（结果：{src['outcome']}）"
                if step_desc not in attempted_steps:
                    attempted_steps.append(step_desc)

    updated_problem: ProblemContext = {
        **state["problem"],
        "attempted_steps": attempted_steps,
    }

    temp_state: SupportState = {
        **state,
        "problem": updated_problem,
        "evidence": merged_evidence,
    }
    projected = project_state_for_supervisor(
        temp_state,
        max_history_messages=deps.settings.max_main_history_messages,
    )
    budget_status = compute_budget_status(temp_state)

    structured_llm = deps.supervisor_llm.with_structured_output(SupervisorAnalysisOutput)
    prompt = [
        SystemMessage(
            content=(
                "你是 SupportFlow 的 Supervisor。请汇总本轮双 Agent / 子任务返回结果：\n"
                "1. 融合文档证据与业务运行证据，判断当前是否足够解答用户问题。\n"
                "2. 显式识别部分失败（timeout/failed）、证据冲突或信息缺口。\n"
                "3. 给出客观评估（不要把任务超时当成正常无异常）。"
            )
        ),
        HumanMessage(
            content=(
                f"汇总上下文：\n{wrap_untrusted_data('merge_view', json.dumps(projected, ensure_ascii=False, indent=2))}\n"
                f"预算状态：{json.dumps(budget_status, ensure_ascii=False)}"
            )
        ),
    ]
    analysis = await structured_llm.ainvoke(prompt)
    assert isinstance(analysis, SupervisorAnalysisOutput)

    final_problem = _merge_problem_context(
        updated_problem,
        analysis,
        verified_account_id=deps.verified_account_id,
    )
    assessment: Assessment = {
        "summary": analysis.summary,
        "evidence_sufficient": analysis.evidence_sufficient,
        "information_gaps": list(analysis.information_gaps),
        "conflicts": [
            {"description": c.description, "evidence_ids": list(c.evidence_ids)}
            for c in analysis.conflicts
        ],
        "tool_health_notes": _collect_tool_health_notes(temp_state),
    }

    return {
        "problem": final_problem,
        "evidence": merged_evidence,
        "assessment": assessment,
        "clarification_questions": (
            list(analysis.clarification_questions) if analysis.requires_user_input else []
        ),
    }


async def decide_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """DECIDE 节点：依据证据充分性、用户意图与全局预算决定下一步流向。"""
    del runtime
    assessment = state["assessment"]
    budget_status = compute_budget_status(state)
    wants_ticket = _user_wants_ticket(state)
    evidence_sufficient = assessment["evidence_sufficient"] if assessment else False

    if wants_ticket and (evidence_sufficient or budget_status["exhausted"]):
        decision: Decision = {
            "action": "draft_ticket",
            "reason": "用户请求创建工单，且当前已完成诊断或预算已尽，进入工单草稿流程",
        }
        return {"decision": decision}

    if evidence_sufficient:
        decision = {
            "action": "answer",
            "reason": "已有充分证据支撑排查结论，输出最终建议",
        }
        return {"decision": decision}

    if budget_status["exhausted"]:
        decision = {
            "action": "answer",
            "reason": "全局调查预算（轮次/工具调用/耗时）已耗尽，输出已有证据与不确定性说明",
        }
        return {"decision": decision}

    if state["clarification_questions"]:
        decision = {
            "action": "ask_user",
            "reason": "仍缺少必要排查条件，向用户追问补充",
        }
        return {"decision": decision}

    decision = {
        "action": "investigate",
        "reason": "证据尚不充分且剩余预算充足，继续下一轮调查",
    }
    return {"decision": decision}


async def answer_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """ANSWER 节点：输出带引用排查建议，并执行确定性引用过滤与不确定性兜底强校验。"""
    deps = runtime.context
    projected = project_state_for_supervisor(
        state,
        max_history_messages=deps.settings.max_main_history_messages,
    )

    structured_llm = deps.supervisor_llm.with_structured_output(AnswerCompositionOutput)
    prompt = [
        SystemMessage(
            content=(
                "你是 SupportFlow 的答复生成专家。请根据已收集证据输出排查建议，包含：\n"
                "1. 当前结论与支撑证据编号（citation_evidence_ids 必须严格来自证据目录）；\n"
                "2. 建议用户执行的操作步骤（严禁把尚未执行的建议写成已执行）；\n"
                "3. 执行摘要（execution_summary）；\n"
                "4. 若存在超时、失败、预算耗尽或信息缺口，必须在 uncertainties 中如实说明。"
            )
        ),
        HumanMessage(
            content=wrap_untrusted_data(
                "answer_context",
                json.dumps(projected, ensure_ascii=False, indent=2),
            )
        ),
    ]
    composed = await structured_llm.ainvoke(prompt)
    assert isinstance(composed, AnswerCompositionOutput)

    # 纯规则检查 1：严格过滤 citation_evidence_ids，剔除任何不存在于 state["evidence"] 的非法引用
    valid_citations = [
        eid
        for eid in dict.fromkeys(composed.citation_evidence_ids)
        if eid in state["evidence"]
    ]

    # 纯规则检查 2：检查是否存在 timeout/failed/budget_exhausted 任务或 information_gaps
    abnormal_notes: list[str] = []
    for round_data in state["rounds"].values():
        for res in round_data["results"] or []:
            if res["status"] in ("timeout", "failed", "budget_exhausted"):
                abnormal_notes.append(
                    f"任务 {res['task_id']}（{res['agent']}）未完整完成（状态={res['status']}）：{res['summary']}"
                )

    assessment = state["assessment"]
    gaps = list(assessment["information_gaps"]) if assessment else []
    if gaps:
        abnormal_notes.append("仍存在未核实的信息缺口：" + "；".join(gaps))

    uncertainties = [u.strip() for u in composed.uncertainties if u.strip()]
    if abnormal_notes and not uncertainties:
        uncertainties = abnormal_notes

    final_content = composed.content.strip()
    if uncertainties and "不确定" not in final_content and "尚未确认" not in final_content:
        final_content += "\n\n【尚未确认与限制说明】\n" + "\n".join(
            f"- {item}" for item in uncertainties
        )

    answer: Answer = {
        "content": final_content,
        "citation_evidence_ids": valid_citations,
        "execution_summary": composed.execution_summary.strip(),
        "uncertainties": uncertainties,
    }
    return {
        "answer": answer,
        "messages": [AIMessage(content=final_content)],
    }
