from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from supportflow.deps import RuntimeDeps
from supportflow.nodes.investigation import exec_investigation_node
from supportflow.nodes.supervisor import (
    answer_node,
    ask_node,
    decide_node,
    merge_node,
    plan_node,
    sup_node,
)
from supportflow.nodes.ticket import (
    cancel_node,
    draft_node,
    edit_node,
    execute_submission_node,
    prepare_submission_node,
    review_node,
)
from supportflow.state import SupportState


def route_entry(
    state: SupportState,
) -> Literal["execute_submission", "sup"]:
    """入口路由：若针对 unknown 工单发起核对/重试且快照存在，直接路由到 execute_submission。"""
    if (
        state.get("entry_action") == "retry_ticket_submission"
        and state["ticket"]["submission"] is not None
    ):
        return "execute_submission"
    return "sup"


def route_after_sup(state: SupportState) -> Literal["ask", "plan", "decide"]:
    decision = state["decision"]
    if decision is None:
        return "plan"
    if decision["action"] == "ask_user":
        return "ask"
    if decision["action"] == "investigate":
        return "plan"
    return "decide"


def route_after_decide(
    state: SupportState,
) -> Literal["plan", "ask", "answer", "draft"]:
    decision = state["decision"]
    if decision is None:
        return "answer"
    action = decision["action"]
    if action == "investigate":
        return "plan"
    if action == "ask_user":
        return "ask"
    if action == "draft_ticket":
        return "draft"
    return "answer"


def route_after_review(
    state: SupportState,
) -> Literal["prepare_submission", "edit", "cancel"]:
    review = state["ticket"]["review"]
    if review is None:
        return "cancel"
    if review["action"] == "approve":
        return "prepare_submission"
    if review["action"] == "edit":
        return "edit"
    return "cancel"


def build_support_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """构建完整 13 节点的 SupportFlow 主图并编译。"""
    builder = StateGraph(SupportState, context_schema=RuntimeDeps)

    # 7 个主诊断与交互节点
    builder.add_node("sup", sup_node)
    builder.add_node("ask", ask_node)
    builder.add_node("plan", plan_node)
    builder.add_node("exec", exec_investigation_node)
    builder.add_node("merge", merge_node)
    builder.add_node("decide", decide_node)
    builder.add_node("answer", answer_node)

    # 6 个按 Checkpointer 边界拆分的工单节点
    builder.add_node("draft", draft_node)
    builder.add_node("review", review_node)
    builder.add_node("edit", edit_node)
    builder.add_node("cancel", cancel_node)
    builder.add_node("prepare_submission", prepare_submission_node)
    builder.add_node("execute_submission", execute_submission_node)

    # 连线与条件路由
    builder.add_conditional_edges(START, route_entry)
    builder.add_conditional_edges("sup", route_after_sup)
    builder.add_edge("ask", "sup")
    builder.add_edge("plan", "exec")
    builder.add_edge("exec", "merge")
    builder.add_edge("merge", "decide")
    builder.add_conditional_edges("decide", route_after_decide)
    builder.add_edge("answer", END)

    builder.add_edge("draft", "review")
    builder.add_conditional_edges("review", route_after_review)
    builder.add_edge("edit", "review")
    builder.add_edge("cancel", END)
    builder.add_edge("prepare_submission", "execute_submission")
    builder.add_edge("execute_submission", END)

    return builder.compile(checkpointer=checkpointer)
