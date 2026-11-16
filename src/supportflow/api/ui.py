from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("SUPPORTFLOW_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="SupportFlow · 多 Agent 技术支持与工单工作台",
    layout="wide",
)


def _api_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0)


def _init_ui_state() -> None:
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "active_thread_id" not in st.session_state:
        st.session_state.active_thread_id = None
    if "thread_history" not in st.session_state:
        st.session_state.thread_history = []
    if "case_data" not in st.session_state:
        st.session_state.case_data = None
    if "draft_question" not in st.session_state:
        st.session_state.draft_question = ""


def _refresh_case(base_url: str, thread_id: str) -> None:
    with _api_client(base_url) as client:
        resp = client.get(f"/cases/{thread_id}")
        resp.raise_for_status()
        st.session_state.case_data = resp.json()


def _build_fault_rules_from_sidebar() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if st.session_state.get("fault_perm_timeout_once"):
        rules.append(
            {
                "target": "get_account_permissions",
                "fault_type": "timeout",
                "trigger_on_attempt": 1,
                "delay_ms": 0,
                "detail": "UI 演示：首次权限查询超时（验证自动单次重试）",
            }
        )
    if st.session_state.get("fault_perm_schema_drift"):
        rules.append(
            {
                "target": "get_account_permissions",
                "fault_type": "invalid_output_schema",
                "trigger_on_attempt": None,
                "delay_ms": 0,
                "detail": "UI 演示：服务端权限工具返回契约损坏（触发局部熔断）",
            }
        )
    if st.session_state.get("fault_ticket_unknown_once"):
        rules.append(
            {
                "target": "ticket_commit",
                "fault_type": "db_commit_unknown",
                "trigger_on_attempt": 1,
                "delay_ms": 0,
                "detail": "UI 演示：数据库提交成功但响应确认丢失（触发 UNKNOWN 恢复）",
            }
        )
    return rules


def _render_sidebar() -> tuple[str, str | None]:
    with st.sidebar:
        st.title("SupportFlow 控制台")
        api_url = st.text_input("FastAPI 服务地址", value=st.session_state.api_url)
        st.session_state.api_url = api_url

        try:
            with _api_client(api_url) as client:
                h = client.get("/health").json()
            st.success(f"后端在线 · 模式: `{h.get('runtime_mode')}`")
        except Exception as exc:
            st.error(f"后端未连接：{exc}")

        st.divider()
        st.subheader("会话与租户沙箱")
        if st.button("开启新问题会话", use_container_width=True):
            st.session_state.active_thread_id = None
            st.session_state.case_data = None
            st.rerun()

        if st.session_state.thread_history:
            selected = st.selectbox(
                "历史会话切换",
                options=st.session_state.thread_history,
                index=(
                    st.session_state.thread_history.index(st.session_state.active_thread_id)
                    if st.session_state.active_thread_id in st.session_state.thread_history
                    else 0
                ),
            )
            if selected != st.session_state.active_thread_id and st.button(
                "加载所选会话", use_container_width=True
            ):
                st.session_state.active_thread_id = selected
                _refresh_case(api_url, selected)
                st.rerun()

        acc_choice = st.selectbox(
            "租户核验账户（OP 沙箱绑定）",
            options=["不锁定（允许触发 ASK 追问）", "acc-001", "acc-002", "acc-003", "acc-004"],
        )
        verified_account_id = None if acc_choice.startswith("不锁定") else acc_choice

        st.divider()
        st.subheader("动态故障注入（现场演示）")
        st.checkbox(
            "首次查询权限超时（测单次重试）",
            key="fault_perm_timeout_once",
        )
        st.checkbox(
            "权限工具契约损坏（测熔断与不确定性）",
            key="fault_perm_schema_drift",
        )
        st.checkbox(
            "工单落库后响应超时（测 UNKNOWN 恢复）",
            key="fault_ticket_unknown_once",
        )

        if st.button("应用故障规则并重置计数", use_container_width=True):
            rules = _build_fault_rules_from_sidebar()
            try:
                with _api_client(api_url) as client:
                    client.post("/faults", json={"rules": rules, "reset_counters": True}).raise_for_status()
                st.toast(f"已更新 {len(rules)} 条故障规则并重置计数")
            except Exception as exc:
                st.error(f"更新故障配置失败：{exc}")

        st.divider()
        st.subheader("快捷演示场景")
        if st.button("场景 1：缺少账户触发 ASK 追问", use_container_width=True):
            st.session_state.draft_question = "升级到 v2.3 后，上传文件返回 403，请求 ID 是 req-123，帮我排查。"
        if st.button("场景 2：排查并生成工单（测审核/UNKNOWN）", use_container_width=True):
            st.session_state.draft_question = "账户 acc-001 在 v2.3 上传文件返回 403（req-123），排查原因后请帮我创建工单。"

        return api_url, verified_account_id


def _render_left_column(
    api_url: str,
    verified_account_id: str | None,
) -> None:
    st.header("诊断对话与人工审核")
    case_data: dict[str, Any] | None = st.session_state.case_data

    # 1) 无活跃会话时展示提交新问题表单
    if case_data is None:
        with st.form("create_case_form"):
            question = st.text_area(
                "描述你要排查的技术问题：",
                value=st.session_state.draft_question,
                height=110,
                placeholder="例如：升级到 v2.3 后，上传文件返回 403，请求 ID 是 req-123，帮我排查。",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                product_version = st.text_input("产品版本（可选）", value="")
            with col_b:
                req_ids_raw = st.text_input("请求 ID（逗号分隔，可选）", value="")

            submitted = st.form_submit_button("提交问题并启动诊断", type="primary")
            if submitted and question.strip():
                req_ids = [r.strip() for r in req_ids_raw.split(",") if r.strip()]
                payload = {
                    "question": question.strip(),
                    "verified_account_id": verified_account_id,
                    "product_version": product_version.strip() or None,
                    "request_ids": req_ids,
                    "fault_rules": _build_fault_rules_from_sidebar(),
                }
                with _api_client(api_url) as client:
                    resp = client.post("/cases", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                tid = data["thread_id"]
                st.session_state.active_thread_id = tid
                if tid not in st.session_state.thread_history:
                    st.session_state.thread_history.insert(0, tid)
                st.session_state.case_data = data
                st.session_state.draft_question = ""
                st.rerun()
        return

    thread_id = case_data["thread_id"]
    st.caption(f"会话 Thread: `{thread_id}` | 业务 Case: `{case_data['case_id']}` | 图状态: `{case_data['status']}`")

    # 2) 对话历史区
    for msg in case_data.get("messages") or []:
        role = "user" if "Human" in msg["role"] else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # 3) 中断人工介入区：ASK 追问 或 REVIEW 工单审核
    interrupts = case_data.get("interrupts") or []
    if interrupts:
        intr = interrupts[0]
        itype = intr.get("type")

        if itype == "clarification":
            st.warning("流程已暂停在 **ASK 节点**（等待补充信息）")
            for q in intr.get("questions") or []:
                st.markdown(f"- 追问：{q}")
            with st.form("ask_resume_form"):
                reply_text = st.text_input("补充说明", value="账户是 acc-001")
                reply_acc = st.text_input("明确账户 ID（可选）", value="acc-001")
                if st.form_submit_button("提交补充信息并恢复执行", type="primary"):
                    resume_payload = {
                        "text": reply_text.strip() or f"账户是 {reply_acc}",
                        "account_id": reply_acc.strip() or None,
                    }
                    with _api_client(api_url) as client:
                        resp = client.post(
                            f"/cases/{thread_id}/resume",
                            json={
                                "resume_payload": resume_payload,
                                "verified_account_id": verified_account_id,
                                "fault_rules": _build_fault_rules_from_sidebar(),
                            },
                        )
                        resp.raise_for_status()
                        st.session_state.case_data = resp.json()
                    st.rerun()

        elif itype == "ticket_review":
            draft = intr.get("draft") or {}
            draft_id = intr.get("draft_id")
            draft_version = intr.get("draft_version")

            st.info(f"流程已暂停在 **REVIEW 节点** · 工单草稿 `{draft_id}` (版本 `v{draft_version}`)")
            with st.container(border=True):
                st.markdown(f"**标题：** {draft.get('title')}")
                st.markdown(f"**详细描述：**\n\n{draft.get('description')}")
                st.markdown(f"**关联证据：** `{draft.get('evidence_ids')}`")
                st.markdown(f"**已执行排查步骤：** `{draft.get('attempted_steps')}`")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("批准并创建工单 (approve)", type="primary", use_container_width=True):
                    with _api_client(api_url) as client:
                        resp = client.post(
                            f"/cases/{thread_id}/resume",
                            json={
                                "resume_payload": {
                                    "draft_id": draft_id,
                                    "draft_version": draft_version,
                                    "action": "approve",
                                },
                                "verified_account_id": verified_account_id,
                                "fault_rules": _build_fault_rules_from_sidebar(),
                            },
                        )
                        resp.raise_for_status()
                        st.session_state.case_data = resp.json()
                    st.rerun()
            with col2:
                if st.button("取消创建 (cancel)", use_container_width=True):
                    with _api_client(api_url) as client:
                        resp = client.post(
                            f"/cases/{thread_id}/resume",
                            json={
                                "resume_payload": {
                                    "draft_id": draft_id,
                                    "draft_version": draft_version,
                                    "action": "cancel",
                                },
                                "verified_account_id": verified_account_id,
                            },
                        )
                        resp.raise_for_status()
                        st.session_state.case_data = resp.json()
                    st.rerun()

            with st.form("edit_draft_form"):
                edit_req = st.text_input("修改意见（提交后将生成新版本草稿）", placeholder="例如：请在标题注明紧急")
                if st.form_submit_button("提交修改意见 (edit)"):
                    if not edit_req.strip():
                        st.error("修改意见不能为空")
                    else:
                        with _api_client(api_url) as client:
                            resp = client.post(
                                f"/cases/{thread_id}/resume",
                                json={
                                    "resume_payload": {
                                        "draft_id": draft_id,
                                        "draft_version": draft_version,
                                        "action": "edit",
                                        "edit_request": edit_req.strip(),
                                    },
                                    "verified_account_id": verified_account_id,
                                },
                            )
                            resp.raise_for_status()
                            st.session_state.case_data = resp.json()
                        st.rerun()

    # 4) 工单提交状态面板（含 UNKNOWN 核对或重试直达入口）
    ticket = case_data.get("ticket") or {}
    t_status = ticket.get("status")
    submission = ticket.get("submission")

    if t_status == "unknown":
        st.error(
            f"工单提交处于 **UNKNOWN 待核实状态**：{ticket.get('error_message')}\n\n"
            f"已保留原幂等键：`{submission.get('idempotency_key') if submission else ''}`"
        )
        if st.button("核对或重试（直达 execute_submission 节点）", type="primary"):
            with _api_client(api_url) as client:
                resp = client.post(
                    f"/cases/{thread_id}/ticket/retry",
                    json={"verified_account_id": verified_account_id},
                )
                resp.raise_for_status()
                st.session_state.case_data = resp.json()
            st.rerun()
    elif t_status == "succeeded":
        st.success(
            f"工单处理完成：ID **`{ticket.get('ticket_id')}`** · "
            f"结果类型 `{ticket.get('result_kind')}` · "
            f"幂等键 `{submission.get('idempotency_key') if submission else ''}`"
        )
    elif t_status in ("conflict", "failed", "cancelled"):
        st.warning(f"工单状态：`{t_status}` · {ticket.get('error_message') or ''}")

    # 5) 最终回答结构化详情
    answer = case_data.get("answer")
    if answer:
        with st.expander("查看 Answer 结构化输出（引用与不确定性校验）", expanded=True):
            st.markdown(f"**执行摘要：** {answer.get('execution_summary')}")
            st.markdown(f"**有效引用 ID：** `{answer.get('citation_evidence_ids')}`")
            uncertainties = answer.get("uncertainties") or []
            if uncertainties:
                st.warning("不确定性与限制声明：\n" + "\n".join(f"- {u}" for u in uncertainties))


def _render_right_column(api_url: str) -> None:
    st.header("调查观测与证据溯源")
    case_data: dict[str, Any] | None = st.session_state.case_data

    if case_data:
        bstat = case_data.get("budget_status") or {}
        m1, m2, m3 = st.columns(3)
        m1.metric("已用调查轮次", bstat.get("rounds_used", 0))
        m2.metric("总工具尝试次数", bstat.get("tool_attempts_used", 0))
        m3.metric("累计调查耗时(ms)", bstat.get("investigation_ms_used", 0))

        # 1) 每轮双 Agent 调查结果
        rounds = case_data.get("rounds") or {}
        if rounds:
            st.subheader("轮次执行明细")
            for round_id, rdata in rounds.items():
                with st.expander(f"轮次 `{round_id}` (耗时 {rdata.get('elapsed_ms')}ms)", expanded=True):
                    for res in rdata.get("results") or []:
                        st.markdown(
                            f"- **{res['agent']}** (`{res['task_id']}`) · 状态: `{res['status']}` · "
                            f"调用 `{res['usage']['tool_attempts']}` 次 / `{res['usage']['agent_steps']}` 步 / "
                            f"`{res['usage']['elapsed_ms']}ms`"
                        )
                        st.caption(res.get("summary"))
                        for err in res.get("errors") or []:
                            st.error(f"[{err['kind']}] {err.get('tool_name') or '-'}: {err['message']}")

        # 2) 证据溯源卡片
        evidence = case_data.get("evidence") or {}
        st.subheader(f"已收集证据池 ({len(evidence)} 条)")
        for eid, ev in evidence.items():
            src = ev.get("source") or {}
            with st.container(border=True):
                if src.get("kind") == "document":
                    st.markdown(
                        f"**`{eid}`** · 文档 `{src.get('document_id')}` "
                        f"(`{src.get('document_version')}` / `{src.get('chunk_id')}`)"
                    )
                else:
                    st.markdown(
                        f"**`{eid}`** · 业务工具 `{src.get('tool_name')}` · "
                        f"结果 `{src.get('outcome')}` · `{src.get('queried_at')}`"
                    )
                st.write(ev.get("content"))

    st.divider()
    st.subheader("全局工单存储（幂等验证视图）")
    try:
        with _api_client(api_url) as client:
            tickets = client.get("/tickets").json()
        if tickets:
            st.dataframe(
                [
                    {
                        "id": t["id"],
                        "account_id": t["account_id"],
                        "title": t["title"],
                        "idempotency_key": t["idempotency_key"],
                        "status": t["status"],
                    }
                    for t in tickets
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("当前暂无已落库工单。")
    except Exception as exc:
        st.caption(f"无法读取工单列表：{exc}")


def main() -> None:
    _init_ui_state()
    api_url, verified_account_id = _render_sidebar()
    col_left, col_right = st.columns([1.2, 1.0])
    with col_left:
        _render_left_column(api_url, verified_account_id)
    with col_right:
        _render_right_column(api_url)


if __name__ == "__main__":
    main()
