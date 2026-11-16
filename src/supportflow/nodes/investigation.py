from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from supportflow.agents.runner import SubAgentRunner
from supportflow.context_policy import build_agent_initial_messages
from supportflow.deps import RuntimeDeps
from supportflow.state import (
    DiagnosisResult,
    Evidence,
    InvestigationRound,
    ProblemContext,
    RoundPlan,
    SupportState,
    TaskPlan,
    create_agent_local_state,
)


async def run_wrapped(
    *,
    round_id: str,
    task: TaskPlan,
    problem: ProblemContext,
    known_evidence: dict[str, Evidence],
    deps: RuntimeDeps,
    round_deadline: float,
) -> DiagnosisResult:
    """WRAP 任务包装器（第 48 点）：从启动即施加超时边界，将超时/异常统一转为 DiagnosisResult。"""
    started_at = deps.clock.monotonic()
    task_deadline = started_at + task["timeout_ms"] / 1000.0
    effective_deadline = min(task_deadline, round_deadline)
    timeout_seconds = max(0.001, effective_deadline - started_at)

    initial_messages = build_agent_initial_messages(
        agent=task["agent"],
        objective=task["objective"],
        problem=problem,
        known_evidence=known_evidence,
        verified_account_id=deps.verified_account_id,
    )

    local = create_agent_local_state(
        round_id=round_id,
        task=task,
        initial_messages=initial_messages,
        deadline_monotonic=effective_deadline,
    )

    runner = SubAgentRunner(deps)
    result: DiagnosisResult

    try:
        async with asyncio.timeout(timeout_seconds):
            result = await runner.run_task(local)
    except TimeoutError:
        local["errors"].append(
            {
                "kind": "timeout",
                "message": f"任务 {task['task_id']} 达到超时期限 ({task['timeout_ms']}ms)",
                "tool_name": None,
                "logical_call_id": None,
            }
        )
        ev_list = list(local["evidence"].values())
        summary = (
            f"任务超时退出，已保留超时前收集的 {len(ev_list)} 条部分证据"
            if ev_list
            else "任务在收集到有效证据前超时"
        )
        result = {
            "round_id": round_id,
            "task_id": task["task_id"],
            "agent": task["agent"],
            "status": "timeout",
            "summary": summary,
            "evidence": ev_list,
            "errors": list(local["errors"]),
            "usage": dict(local["usage"]),
        }
    except asyncio.CancelledError:
        # 保留外部对整个图执行的显式取消信号，不吞掉 CancelledError
        raise
    except Exception as exc:
        local["errors"].append(
            {
                "kind": "unhandled",
                "message": f"任务未捕获异常: {type(exc).__name__}: {exc}",
                "tool_name": None,
                "logical_call_id": None,
            }
        )
        result = {
            "round_id": round_id,
            "task_id": task["task_id"],
            "agent": task["agent"],
            "status": "failed",
            "summary": f"任务执行异常退出（{type(exc).__name__}），已保留 {len(local['evidence'])} 条局部证据",
            "evidence": list(local["evidence"].values()),
            "errors": list(local["errors"]),
            "usage": dict(local["usage"]),
        }

    elapsed_ms = max(0, int((deps.clock.monotonic() - started_at) * 1000))
    # 使用执行器权威标识与真实耗时覆盖结果，不信任模型自报计数
    return {
        "round_id": round_id,
        "task_id": task["task_id"],
        "agent": task["agent"],
        "status": result["status"],
        "summary": result["summary"],
        "evidence": result["evidence"],
        "errors": result["errors"],
        "usage": {
            "tool_attempts": local["usage"]["tool_attempts"],
            "agent_steps": local["usage"]["agent_steps"],
            "elapsed_ms": elapsed_ms,
        },
    }


async def execute_round(
    *,
    plan: RoundPlan,
    problem: ProblemContext,
    known_evidence: dict[str, Evidence],
    deps: RuntimeDeps,
) -> tuple[list[DiagnosisResult], int]:
    """DISPATCH -> COLLECT -> OUTPUT：并发启动本轮所有任务，等待全部返回并校验一一对应。"""
    round_started_at = deps.clock.monotonic()
    round_deadline = round_started_at + plan["timeout_ms"] / 1000.0

    handles: dict[str, asyncio.Task[DiagnosisResult]] = {}
    for task in plan["tasks"]:
        handles[task["task_id"]] = asyncio.create_task(
            run_wrapped(
                round_id=plan["round_id"],
                task=task,
                problem=problem,
                known_evidence=known_evidence,
                deps=deps,
                round_deadline=round_deadline,
            )
        )

    raw_results = await asyncio.gather(*handles.values())

    expected_ids = [t["task_id"] for t in plan["tasks"]]
    result_by_id: dict[str, DiagnosisResult] = {}
    for res in raw_results:
        if res["round_id"] != plan["round_id"]:
            raise RuntimeError(
                f"round_id 不一致: expected={plan['round_id']}, got={res['round_id']}"
            )
        result_by_id[res["task_id"]] = res

    if set(result_by_id.keys()) != set(expected_ids):
        raise RuntimeError(
            f"任务结果集合与计划不匹配: expected={expected_ids}, got={list(result_by_id.keys())}"
        )

    ordered_results = [result_by_id[tid] for tid in expected_ids]
    round_elapsed_ms = max(0, int((deps.clock.monotonic() - round_started_at) * 1000))
    return ordered_results, round_elapsed_ms


async def exec_investigation_node(
    state: SupportState,
    runtime: Runtime[RuntimeDeps],
) -> dict[str, Any]:
    """主图 EXEC 节点：本轮并发执行结束后一次性将结果写回主图 State。"""
    round_id = state["current_round_id"]
    if round_id is None or round_id not in state["rounds"]:
        raise ValueError(f"无效的 current_round_id: {round_id}")

    round_data = state["rounds"][round_id]
    results, elapsed_ms = await execute_round(
        plan=round_data["plan"],
        problem=state["problem"],
        known_evidence=state["evidence"],
        deps=runtime.context,
    )

    updated_round: InvestigationRound = {
        "plan": round_data["plan"],
        "results": results,
        "elapsed_ms": elapsed_ms,
    }
    return {
        "rounds": {
            **state["rounds"],
            round_id: updated_round,
        }
    }
