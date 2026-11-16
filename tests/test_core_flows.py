from __future__ import annotations

import pytest
from langgraph.types import Command

from supportflow.env_factory import SimulationChatModel
from supportflow.nodes.investigation import execute_round
from supportflow.nodes.supervisor import compute_budget_status
from supportflow.schemas import FaultInjectorConfig, FaultRule
from supportflow.state import RoundPlan, create_initial_state


@pytest.mark.asyncio
async def test_e2e_dual_agent_with_ask_interrupt_and_resume(memory_graph, make_deps, sim_settings):
    """覆盖：正常双 Agent 并行诊断 + 缺少账户时在 ASK 暂停追问，补充后恢复并输出有效引用。"""
    deps = make_deps()
    config = {"configurable": {"thread_id": "thread-e2e-ask"}}
    initial = create_initial_state(
        "升级到 v2.3 后，上传文件返回 403，请求 ID 是 req-123，帮我排查。",
        sim_settings.to_budget_policy(),
    )

    # 1) 首次运行因缺少 account_id 在 ASK 节点触发 interrupt
    first_out = await memory_graph.ainvoke(initial, config=config, context=deps)
    assert "__interrupt__" in first_out
    interrupt_payload = first_out["__interrupt__"][0].value
    assert interrupt_payload["type"] == "clarification"
    assert first_out["problem"]["account_id"] is None

    # 2) 用户补充账户 acc-001 后恢复执行，走完 SUP -> PLAN -> EXEC(双Agent并发) -> MERGE -> DECIDE -> ANSWER
    final_out = await memory_graph.ainvoke(
        Command(resume={"text": "待排查账户是 acc-001", "account_id": "acc-001"}),
        config=config,
        context=deps,
    )
    assert "__interrupt__" not in final_out
    assert final_out["problem"]["account_id"] == "acc-001"
    assert final_out["answer"] is not None

    answer = final_out["answer"]
    assert "storage:write" in answer["content"]
    assert len(answer["citation_evidence_ids"]) >= 2
    for eid in answer["citation_evidence_ids"]:
        assert eid in final_out["evidence"]

    # 确认双 Agent 并行产物均已写入主图（含 doc 与 biz 证据）
    kinds = {ev["source"]["kind"] for ev in final_out["evidence"].values()}
    assert kinds == {"document", "business"}

    # 校验预算口径一致性
    budget_status = compute_budget_status(final_out)
    assert budget_status["rounds_used"] == 1
    assert budget_status["tool_attempts_used"] >= 3


@pytest.mark.asyncio
async def test_transient_timeout_retries_once_and_blocks_cross_loop_replay(make_deps, sim_settings):
    """覆盖：瞬时超时自动内部重试 1 次成功；以及持续超时时跨 Agent 循环严守最多重试 1 次上限。"""
    # 场景 A：第 1 次超时、第 2 次恢复正常（不额外调 LLM，直接原参重试成功）
    deps_recover = make_deps(
        fault_config=FaultInjectorConfig(
            rules=[
                FaultRule(
                    target="get_account_permissions",
                    fault_type="timeout",
                    trigger_on_attempt=1,
                    detail="首次权限查询网络抖动",
                )
            ]
        ),
        verified_account_id="acc-001",
    )
    state_a = create_initial_state(
        "账户 acc-001 在 v2.3 上传 403，req-123",
        sim_settings.to_budget_policy(),
        account_id="acc-001",
        product_version="v2.3",
        request_ids=["req-123"],
    )
    plan_a: RoundPlan = {
        "round_id": "r-retry-ok",
        "timeout_ms": 10000,
        "tasks": [
            {
                "task_id": "r-retry-ok:operations",
                "agent": "operations",
                "objective": "检查 acc-001 的上传权限与 req-123 日志",
                "max_tool_attempts": 5,
                "max_steps": 5,
                "timeout_ms": 10000,
            }
        ],
    }
    results_a, _ = await execute_round(
        plan=plan_a,
        problem=state_a["problem"],
        known_evidence={},
        deps=deps_recover,
    )
    res_a = results_a[0]
    assert res_a["status"] == "completed"
    # 权限(首次超时+重试成功=2次) + 日志(1次) = 3 次尝试
    assert res_a["usage"]["tool_attempts"] >= 3
    assert any(
        e["kind"] == "timeout" and e["tool_name"] == "get_account_permissions"
        for e in res_a["errors"]
    )
    assert any(
        ev["source"]["kind"] == "business"
        and ev["source"]["tool_name"] == "get_account_permissions"
        for ev in res_a["evidence"]
    )

    # 场景 B：持续超时 + 诱导模型跨循环再调同一逻辑调用，验证 logical_calls 死守 2 次（1首次+1重试）
    llm_repeat = SimulationChatModel(force_repeat_tool_call="get_account_permissions")
    deps_persistent = make_deps(
        llm=llm_repeat,
        fault_config=FaultInjectorConfig(
            rules=[
                FaultRule(
                    target="get_account_permissions",
                    fault_type="timeout",
                    trigger_on_attempt=None,
                    detail="持续超时",
                )
            ]
        ),
        verified_account_id="acc-001",
    )
    plan_b: RoundPlan = {
        "round_id": "r-retry-cap",
        "timeout_ms": 10000,
        "tasks": [
            {
                "task_id": "r-retry-cap:operations",
                "agent": "operations",
                "objective": "检查 acc-001 上传权限",
                "max_tool_attempts": 6,
                "max_steps": 6,
                "timeout_ms": 10000,
            }
        ],
    }
    results_b, _ = await execute_round(
        plan=plan_b,
        problem=state_a["problem"],
        known_evidence={},
        deps=deps_persistent,
    )
    res_b = results_b[0]
    perm_timeout_errors = [
        e
        for e in res_b["errors"]
        if e["kind"] == "timeout" and e["tool_name"] == "get_account_permissions"
    ]
    # 第1步：1次首次 + 1次内部重试 = 2次超时；第2步跨循环重放被 BUDGET/RECORD 直接拦截
    assert len(perm_timeout_errors) == 2
    assert any(
        e["kind"] == "tool_error" and "已达到尝试/重试上限" in e["message"]
        for e in res_b["errors"]
    )


@pytest.mark.asyncio
async def test_output_validation_circuit_breaker_and_answer_uncertainty_fallback(
    memory_graph,
    make_deps,
):
    """覆盖：服务端契约损坏立即熔断禁用（不瞎改参数重调）+ ANSWER 纯规则强制补齐 uncertainties。"""
    from supportflow.config import Settings

    tight_settings = Settings(
        runtime_mode="simulation",
        default_max_rounds=1,
        default_max_tool_attempts=6,
        default_max_investigation_ms=20000,
        default_max_agent_steps_per_task=5,
        default_round_timeout_ms=10000,
        default_task_timeout_ms=10000,
        default_tool_timeout_ms=3000,
    )
    # 故意开启 omit_uncertainties_in_llm=True，测试 ANSWER 的 Python 确定性兜底强校验
    llm = SimulationChatModel(omit_uncertainties_in_llm=True)
    from supportflow.env_factory import create_simulation_deps

    deps = create_simulation_deps(
        settings=tight_settings,
        fault_config=FaultInjectorConfig(
            rules=[
                FaultRule(
                    target="get_account_permissions",
                    fault_type="invalid_output_schema",
                    trigger_on_attempt=None,
                )
            ]
        ),
        llm=llm,
        use_test_embedder=True,
        verified_account_id="acc-001",
    )

    config = {"configurable": {"thread_id": "thread-schema-drift"}}
    initial = create_initial_state(
        "账户 acc-001 在 v2.3 上传 403，帮我排查权限原因",
        tight_settings.to_budget_policy(),
        account_id="acc-001",
        product_version="v2.3",
    )
    out = await memory_graph.ainvoke(initial, config=config, context=deps)

    round1_results = out["rounds"]["%s:r1" % out["case_id"]]["results"]
    assert round1_results is not None
    ops_res = next(r for r in round1_results if r["agent"] == "operations")

    # 校验 get_account_permissions 仅调用 1 次即因 output_validation 被熔断，没有被模型傻傻修改参数重调
    ov_errors = [e for e in ops_res["errors"] if e["kind"] == "output_validation"]
    assert len(ov_errors) == 1
    assert ops_res["usage"]["tool_attempts"] == 1

    # 校验 ANSWER 节点在模型漏写 uncertainties 时，由程序自动补齐非空不确定性声明
    assert out["answer"] is not None
    assert len(out["answer"]["uncertainties"]) >= 1
    assert "尚未确认" in out["answer"]["content"]


@pytest.mark.asyncio
async def test_two_step_no_evidence_early_exit(make_deps, sim_settings):
    """覆盖：连续两步无任何新证据产出时，提前收敛退出（不白烧完 max_steps=6）。"""
    deps = make_deps()
    state = create_initial_state(
        "v9.9 版本上传报错如何处理？",
        sim_settings.to_budget_policy(),
        product_version="v9.9",
    )
    plan: RoundPlan = {
        "round_id": "r-no-ev",
        "timeout_ms": 10000,
        "tasks": [
            {
                "task_id": "r-no-ev:knowledge",
                "agent": "knowledge",
                "objective": "检索 v9.9 文档说明",
                "max_tool_attempts": 5,
                "max_steps": 6,
                "timeout_ms": 10000,
            }
        ],
    }
    results, _ = await execute_round(
        plan=plan,
        problem=state["problem"],
        known_evidence={},
        deps=deps,
    )
    res = results[0]
    assert res["status"] == "no_answer"
    assert res["usage"]["agent_steps"] == 2
    assert "连续两步调查未产出新证据" in res["summary"]


@pytest.mark.asyncio
async def test_op_tenant_sandbox_rejects_cross_account_access(make_deps, sim_settings):
    """覆盖：OP 节点租户沙箱拦截跨账户越权查询（不消耗 tool_attempts）并引导模型自纠。"""
    llm = SimulationChatModel(wrong_account_first_call="acc-999")
    deps = make_deps(llm=llm, verified_account_id="acc-001")
    state = create_initial_state(
        "排查账户 acc-001 的上传权限问题",
        sim_settings.to_budget_policy(),
        account_id="acc-001",
        product_version="v2.3",
    )
    plan: RoundPlan = {
        "round_id": "r-sandbox",
        "timeout_ms": 10000,
        "tasks": [
            {
                "task_id": "r-sandbox:operations",
                "agent": "operations",
                "objective": "查询账户权限",
                "max_tool_attempts": 3,
                "max_steps": 5,
                "timeout_ms": 10000,
            }
        ],
    }
    results, _ = await execute_round(
        plan=plan,
        problem=state["problem"],
        known_evidence={},
        deps=deps,
    )
    res = results[0]
    assert res["status"] == "completed"
    # 第 1 步越权被拦截记入 argument_validation，不计入 tool_attempts；第 2 步纠偏为 acc-001 成功调用 1 次
    assert res["usage"]["tool_attempts"] == 1
    assert any(
        e["kind"] == "argument_validation" and "租户沙箱拦截" in e["message"]
        for e in res["errors"]
    )
    assert any("acc-001" in ev["content"] for ev in res["evidence"])
