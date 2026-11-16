from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from supportflow.context_policy import project_state_for_supervisor, wrap_untrusted_data
from supportflow.deps import RuntimeDeps
from supportflow.schemas import DraftTicketOutput, TicketPayload, TicketReviewAction
from supportflow.state import SupportState, TicketDraft, TicketReview, TicketState, TicketSubmission


def _update_ticket(state: SupportState, **updates: Any) -> dict[str, Any]:
    merged: TicketState = {**state["ticket"], **updates}
    return {"ticket": merged}


async def draft_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """DRAFT 节点：基于已核实证据与实际已执行步骤生成第 1 版工单草稿。"""
    deps = runtime.context
    projected = project_state_for_supervisor(
        state,
        max_history_messages=deps.settings.max_main_history_messages,
    )

    structured_llm = deps.supervisor_llm.with_structured_output(DraftTicketOutput)
    prompt = [
        SystemMessage(
            content=(
                "请将当前排查结果整理为技术支持工单草稿（title、description、evidence_ids、attempted_steps）。"
                "要求：evidence_ids 必须来自已有证据；attempted_steps 只填写已实际执行的排查动作。"
            )
        ),
        HumanMessage(
            content=wrap_untrusted_data(
                "draft_context",
                json.dumps(projected, ensure_ascii=False, indent=2),
            )
        ),
    ]
    draft_out = await structured_llm.ainvoke(prompt)
    assert isinstance(draft_out, DraftTicketOutput)

    valid_eids = [
        eid for eid in dict.fromkeys(draft_out.evidence_ids) if eid in state["evidence"]
    ]
    if not valid_eids and state["evidence"]:
        valid_eids = list(state["evidence"].keys())

    # 严格从 state["problem"]["attempted_steps"] 兜底，防止把建议步骤写进已执行步骤
    real_attempted = list(state["problem"]["attempted_steps"])
    if not real_attempted and draft_out.attempted_steps:
        real_attempted = [
            s for s in draft_out.attempted_steps if s in state["problem"]["attempted_steps"]
        ]

    draft: TicketDraft = {
        "draft_id": f"draft-{uuid.uuid4().hex[:10]}",
        "version": 1,
        "title": draft_out.title.strip(),
        "description": draft_out.description.strip(),
        "evidence_ids": valid_eids,
        "attempted_steps": real_attempted,
    }
    return _update_ticket(
        state,
        status="draft",
        draft=draft,
        review=None,
        submission=None,
        ticket_id=None,
        result_kind=None,
        error_message=None,
    )


async def review_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """REVIEW 节点：通过 interrupt 暂停并等待用户审核（approve / edit / cancel），严格校验草稿版本号。"""
    del runtime
    draft = state["ticket"]["draft"]
    if draft is None:
        raise ValueError("进入 review 节点前必须先生成工单草稿")

    raw_response = interrupt(
        {
            "type": "ticket_review",
            "case_id": state["case_id"],
            "draft_id": draft["draft_id"],
            "draft_version": draft["version"],
            "draft": draft,
            "actions": ["approve", "edit", "cancel"],
        }
    )

    action_model = TicketReviewAction.model_validate(raw_response)
    if action_model.draft_id != draft["draft_id"]:
        raise ValueError(
            f"工单草稿 ID 不匹配: expected={draft['draft_id']}, got={action_model.draft_id}"
        )
    if action_model.draft_version != draft["version"]:
        raise ValueError(
            f"工单草稿版本号已过期: expected=v{draft['version']}, got=v{action_model.draft_version}"
        )

    review: TicketReview = {
        "draft_id": action_model.draft_id,
        "draft_version": action_model.draft_version,
        "action": action_model.action,
        "edit_request": action_model.edit_request,
    }
    next_status = "approved" if action_model.action == "approve" else state["ticket"]["status"]
    return _update_ticket(state, review=review, status=next_status)


async def edit_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """EDIT 节点：按用户修改意见生成新版本草稿（version + 1），并清空旧审核结果。"""
    deps = runtime.context
    old_draft = state["ticket"]["draft"]
    review = state["ticket"]["review"]
    if old_draft is None or review is None or not review["edit_request"]:
        raise ValueError("edit 节点缺少草稿或修改意见")

    structured_llm = deps.supervisor_llm.with_structured_output(DraftTicketOutput)
    prompt = [
        SystemMessage(
            content=(
                "请根据用户的修改意见更新工单草稿，保留真实证据编号与已实际执行的排查步骤。"
            )
        ),
        HumanMessage(
            content=(
                f"原草稿 (v{old_draft['version']})：\n"
                f"{wrap_untrusted_data('old_draft', json.dumps(old_draft, ensure_ascii=False, indent=2))}\n\n"
                f"用户修改意见：\n"
                f"{wrap_untrusted_data('edit_request', review['edit_request'])}"
            )
        ),
    ]
    updated_out = await structured_llm.ainvoke(prompt)
    assert isinstance(updated_out, DraftTicketOutput)

    valid_eids = [
        eid for eid in dict.fromkeys(updated_out.evidence_ids) if eid in state["evidence"]
    ] or list(old_draft["evidence_ids"])

    new_draft: TicketDraft = {
        "draft_id": old_draft["draft_id"],
        "version": old_draft["version"] + 1,
        "title": updated_out.title.strip(),
        "description": updated_out.description.strip(),
        "evidence_ids": valid_eids,
        "attempted_steps": list(old_draft["attempted_steps"]),
    }
    return _update_ticket(
        state,
        status="draft",
        draft=new_draft,
        review=None,
    )


async def cancel_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """CANCEL 节点：取消尚未提交的工单草稿流程。"""
    del runtime
    return {
        **_update_ticket(state, status="cancelled"),
        "messages": [AIMessage(content="已根据您的指令取消工单创建。之前的诊断证据仍保留在会话中。")],
    }


async def prepare_submission_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """prepare_submission 节点（REQUEST）：固定提交快照与稳定幂等键，并在写库前由 Checkpointer 落盘。"""
    ticket = state["ticket"]
    draft = ticket["draft"]
    review = ticket["review"]
    if draft is None or review is None:
        raise ValueError("prepare_submission 缺少草稿或审核记录")
    if review["action"] != "approve" or review["draft_version"] != draft["version"]:
        raise ValueError("仅当最新草稿被明确 approve 后才允许准备提交快照")

    # 幂等保护：若已为当前确认版本生成过 submission 快照，直接复用，绝不重新生成新键
    existing_sub = ticket["submission"]
    if (
        existing_sub is not None
        and existing_sub["approved_draft_id"] == draft["draft_id"]
        and existing_sub["approved_draft_version"] == draft["version"]
    ):
        return {}

    account_id = (
        runtime.context.verified_account_id
        or state["problem"]["account_id"]
        or "acc-unknown"
    )
    payload_model = TicketPayload(
        case_id=state["case_id"],
        account_id=account_id,
        title=draft["title"],
        description=draft["description"],
        evidence_ids=list(draft["evidence_ids"]),
        attempted_steps=list(draft["attempted_steps"]),
        product_version=state["problem"]["product_version"],
        error_codes=list(state["problem"]["error_codes"]),
        request_ids=list(state["problem"]["request_ids"]),
    )
    payload_json = json.dumps(
        payload_model.model_dump(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    idempotency_key = f"idem:{state['case_id']}:{draft['draft_id']}:v{draft['version']}"

    submission: TicketSubmission = {
        "idempotency_key": idempotency_key,
        "approved_draft_id": draft["draft_id"],
        "approved_draft_version": draft["version"],
        "payload_json": payload_json,
        "payload_hash": payload_hash,
    }
    return _update_ticket(
        state,
        status="prepared",
        submission=submission,
        error_message=None,
    )


async def execute_submission_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """execute_submission 节点（LOOKUP + TX + COMMIT）：使用已落盘快照执行幂等建单，并支持 unknown 入口重试。"""
    deps = runtime.context
    submission = state["ticket"]["submission"]
    if submission is None:
        raise ValueError("execute_submission 缺少已持久化的 TicketSubmission 快照")

    account_id = (
        deps.verified_account_id
        or state["problem"]["account_id"]
        or "acc-unknown"
    )
    outcome = await deps.ticket_store.create_idempotent(submission, account_id)

    if outcome.outcome == "created":
        assert outcome.ticket is not None
        msg = f"工单已创建成功：{outcome.ticket.id}（标题：{outcome.ticket.title}）。"
        return {
            **_update_ticket(
                state,
                status="succeeded",
                ticket_id=outcome.ticket.id,
                result_kind="created",
                error_message=None,
            ),
            "entry_action": "diagnose",
            "messages": [AIMessage(content=msg)],
        }

    if outcome.outcome == "existing":
        assert outcome.ticket is not None
        msg = f"核对命中已创建工单：{outcome.ticket.id}（幂等返回原工单，未重复创建）。"
        return {
            **_update_ticket(
                state,
                status="succeeded",
                ticket_id=outcome.ticket.id,
                result_kind="existing",
                error_message=None,
            ),
            "entry_action": "diagnose",
            "messages": [AIMessage(content=msg)],
        }

    if outcome.outcome == "conflict":
        err = outcome.error_message or "同一幂等键存在内容冲突"
        return {
            **_update_ticket(
                state,
                status="conflict",
                error_message=err,
            ),
            "entry_action": "diagnose",
            "messages": [AIMessage(content=f"工单提交冲突：{err}")],
        }

    if outcome.outcome == "unknown":
        err = outcome.error_message or "提交结果待核实"
        return {
            **_update_ticket(
                state,
                status="unknown",
                error_message=err,
            ),
            "entry_action": "diagnose",
            "messages": [
                AIMessage(
                    content=(
                        f"工单提交结果暂时不确定（{err}）。"
                        "原提交快照与幂等键已安全保留，请点击“核对或重试”查询状态，避免重复新建。"
                    )
                )
            ],
        }

    err = outcome.error_message or "工单提交失败"
    return {
        **_update_ticket(
            state,
            status="failed",
            error_message=err,
        ),
        "entry_action": "diagnose",
        "messages": [AIMessage(content=f"工单提交失败：{err}")],
    }
