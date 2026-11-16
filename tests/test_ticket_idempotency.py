from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from langgraph.types import Command

from supportflow.persistence.ticket_service import InMemoryTicketStore
from supportflow.schemas import FaultInjectorConfig, FaultRule
from supportflow.state import TicketSubmission, create_initial_state
from supportflow.tools import FaultInjector


@pytest.mark.asyncio
async def test_ticket_review_edit_version_guard_and_approve(memory_graph, make_deps, sim_settings):
    """覆盖：草稿生成 -> v1 审核过期版本拦截 -> edit(v2) -> approve -> prepare -> execute 成功建单。"""
    store = InMemoryTicketStore()
    deps = make_deps(ticket_store=store, verified_account_id="acc-001")
    config = {"configurable": {"thread_id": "thread-ticket-edit"}}
    initial = create_initial_state(
        "账户 acc-001 在 v2.3 上传文件返回 403（req-123），排查原因后请帮我创建工单。",
        sim_settings.to_budget_policy(),
        account_id="acc-001",
        product_version="v2.3",
        request_ids=["req-123"],
    )

    # 1) 诊断完成并进入 review 暂停等待审核 (v1)
    paused_v1 = await memory_graph.ainvoke(initial, config=config, context=deps)
    assert "__interrupt__" in paused_v1
    intr_v1 = paused_v1["__interrupt__"][0].value
    assert intr_v1["type"] == "ticket_review"
    draft_id = intr_v1["draft_id"]
    assert intr_v1["draft_version"] == 1

    # 2) 提交过期版本号 (v99)，应被 review 节点拒绝且不影响已有 checkpoint 状态
    with pytest.raises(ValueError, match="版本号已过期"):
        await memory_graph.ainvoke(
            Command(
                resume={
                    "draft_id": draft_id,
                    "draft_version": 99,
                    "action": "approve",
                }
            ),
            config=config,
            context=deps,
        )

    # 3) 对 v1 提出修改意见 -> edit -> 重新停在 v2 审核
    paused_v2 = await memory_graph.ainvoke(
        Command(
            resume={
                "draft_id": draft_id,
                "draft_version": 1,
                "action": "edit",
                "edit_request": "请在标题注明紧急，并补充要求重新签发令牌",
            }
        ),
        config=config,
        context=deps,
    )
    assert "__interrupt__" in paused_v2
    intr_v2 = paused_v2["__interrupt__"][0].value
    assert intr_v2["draft_version"] == 2
    assert "紧急" in intr_v2["draft"]["title"]

    # 4) 确认 v2 -> prepare_submission -> execute_submission -> 成功创建唯一工单
    final_out = await memory_graph.ainvoke(
        Command(
            resume={
                "draft_id": draft_id,
                "draft_version": 2,
                "action": "approve",
            }
        ),
        config=config,
        context=deps,
    )
    assert "__interrupt__" not in final_out
    ticket = final_out["ticket"]
    assert ticket["status"] == "succeeded"
    assert ticket["result_kind"] == "created"
    assert ticket["ticket_id"] is not None
    assert ticket["submission"] is not None
    assert ticket["submission"]["approved_draft_version"] == 2
    assert store.count_tickets() == 1


@pytest.mark.asyncio
async def test_serial_10_and_concurrent_10_idempotent_submissions_and_conflict():
    """覆盖：同键串行 10 次、同键并发 10 次均只创建 1 张工单；同键不同内容返回 conflict。"""
    payload_dict = {
        "case_id": "case-idem-001",
        "account_id": "acc-001",
        "title": "v2.3 上传 403 工单",
        "description": "缺少 storage:write 权限",
        "evidence_ids": ["doc:upload-auth-guide:v2.3:chunk-01"],
        "attempted_steps": ["调用 get_account_permissions"],
    }
    payload_json = json.dumps(payload_dict, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    submission: TicketSubmission = {
        "idempotency_key": "idem:case-idem-001:draft-abc:v1",
        "approved_draft_id": "draft-abc",
        "approved_draft_version": 1,
        "payload_json": payload_json,
        "payload_hash": payload_hash,
    }

    # 1) 同键串行重复提交 10 次
    serial_store = InMemoryTicketStore(simulate_race_yield=False)
    serial_outcomes = [
        await serial_store.create_idempotent(submission, "acc-001") for _ in range(10)
    ]
    assert serial_outcomes[0].outcome == "created"
    assert all(o.outcome == "existing" for o in serial_outcomes[1:])
    serial_ids = {o.ticket.id for o in serial_outcomes if o.ticket is not None}
    assert len(serial_ids) == 1
    assert serial_store.count_tickets() == 1

    # 2) 同键并发提交 10 次（强制在 LOOKUP 后让出协程调度，触发唯一约束冲突回滚重查路径）
    concurrent_store = InMemoryTicketStore(simulate_race_yield=True)
    concurrent_outcomes = await asyncio.gather(
        *(concurrent_store.create_idempotent(submission, "acc-001") for _ in range(10))
    )
    created_cnt = sum(1 for o in concurrent_outcomes if o.outcome == "created")
    existing_cnt = sum(1 for o in concurrent_outcomes if o.outcome == "existing")
    assert created_cnt == 1
    assert existing_cnt == 9
    concurrent_ids = {o.ticket.id for o in concurrent_outcomes if o.ticket is not None}
    assert len(concurrent_ids) == 1
    assert concurrent_store.count_tickets() == 1

    # 3) 同幂等键但修改提交内容 -> 返回 conflict 拒绝覆盖
    tampered_payload = {**payload_dict, "title": "完全不同的下载速度缓慢问题"}
    tampered_json = json.dumps(
        tampered_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    tampered_submission: TicketSubmission = {
        **submission,
        "payload_json": tampered_json,
        "payload_hash": hashlib.sha256(tampered_json.encode("utf-8")).hexdigest(),
    }
    conflict_outcome = await concurrent_store.create_idempotent(tampered_submission, "acc-001")
    assert conflict_outcome.outcome == "conflict"
    assert concurrent_store.count_tickets() == 1


@pytest.mark.asyncio
async def test_db_commit_unknown_recovery_via_entry_direct_route(
    memory_graph,
    make_deps,
    sim_settings,
):
    """覆盖：模拟工单落库后响应丢失进入 unknown 状态，后续通过 entry_action 直达 execute_submission 找回原工单。"""
    fault_cfg = FaultInjectorConfig(
        rules=[
            FaultRule(
                target="ticket_commit",
                fault_type="db_commit_unknown",
                trigger_on_attempt=1,
                detail="数据库提交成功但网络返回确认超时",
            )
        ]
    )
    injector = FaultInjector(fault_cfg)
    store = InMemoryTicketStore(fault_injector=injector, simulate_race_yield=False)
    deps = make_deps(
        fault_config=fault_cfg,
        ticket_store=store,
        verified_account_id="acc-001",
    )
    config = {"configurable": {"thread_id": "thread-ticket-unknown"}}
    initial = create_initial_state(
        "账户 acc-001 在 v2.3 上传 403，req-123，请直接帮我建单处理。",
        sim_settings.to_budget_policy(),
        account_id="acc-001",
        product_version="v2.3",
        request_ids=["req-123"],
    )

    # 1) 诊断完成进入工单审核暂停
    paused = await memory_graph.ainvoke(initial, config=config, context=deps)
    intr = paused["__interrupt__"][0].value
    draft_id = intr["draft_id"]

    # 2) 用户确认后先经过 prepare_submission（持久化快照），再进入 execute_submission 命中 db_commit_unknown
    unknown_out = await memory_graph.ainvoke(
        Command(
            resume={
                "draft_id": draft_id,
                "draft_version": 1,
                "action": "approve",
            }
        ),
        config=config,
        context=deps,
    )
    assert unknown_out["ticket"]["status"] == "unknown"
    assert unknown_out["ticket"]["ticket_id"] is None
    assert unknown_out["ticket"]["submission"] is not None
    assert store.count_tickets() == 1
    saved_key = unknown_out["ticket"]["submission"]["idempotency_key"]

    # 3) 通过入口直达路由重试：不重跑 sup/plan/exec/draft/review，直接进入 execute_submission 核对找回原工单
    recovered_out = await memory_graph.ainvoke(
        {"entry_action": "retry_ticket_submission"},
        config=config,
        context=deps,
    )
    assert "__interrupt__" not in recovered_out
    assert recovered_out["ticket"]["status"] == "succeeded"
    assert recovered_out["ticket"]["result_kind"] == "existing"
    assert recovered_out["ticket"]["ticket_id"] == store.list_tickets()[0].id
    assert recovered_out["ticket"]["submission"]["idempotency_key"] == saved_key
    assert recovered_out["entry_action"] == "diagnose"
    # 关键：全过程仅物理创建了 1 张工单，未重复新建
    assert store.count_tickets() == 1
