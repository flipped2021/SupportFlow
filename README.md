# SupportFlow｜多 Agent 技术支持与工单工作台

面向模拟文件存储 SaaS 场景构建的多 Agent 技术支持系统，基于 **LangGraph + FastAPI + PostgreSQL/pgvector + BGE-M3 + BM25 + Streamlit** 实现，聚焦**并发子 Agent 隔离编排、跨循环有限容错、上下文治理与工单强幂等事务**。

---

## 核心架构特性

1. **13 节点主图与 Checkpointer 断点恢复（`src/supportflow/graph.py`）**
   * 主图节点：`sup`、`ask`（`interrupt`）、`plan`、`exec`、`merge`、`decide`、`answer`、`draft`、`review`（`interrupt`）、`edit`、`cancel`、`prepare_submission`、`execute_submission`。
   * `prepare_submission` 与 `execute_submission` 严格拆分为两个独立节点，利用 LangGraph 节点边界确保 `idempotency_key` 与 `payload_hash` 先落盘到 Checkpoint，再执行数据库事务；当处于 `unknown` 状态发起“核对或重试”时，图入口（`route_entry`）直接路由至 `execute_submission` 复用原快照 `LOOKUP`。
2. **并发子 Agent 执行与包装器隔离（`nodes/investigation.py` + `agents/runner.py`）**
   * `EXEC` 节点内并发启动 `Knowledge` 与 `Operations` 子 Agent，各自维护独立 `AgentLocalState`，统一由 `run_wrapped` 施加任务与轮次硬超时并收敛为结构化 `DiagnosisResult`。
   * 基于 `make_logical_call_id` 在 `AgentLocalState` 维护跨循环 `LogicalCallRecord`，运行诊断对 `TransientToolError` 最多自动原地重试 1 次，重新进入循环不重置计数。
3. **纯规则验证闭环与上下文治理（`context_policy.py` + `nodes/supervisor.py`）**
   * 角色上下文投影、单次工具返回截断（`max_tool_output_chars`）、不可信证据定界标签隔离；
   * 严格区分 `argument_validation`（回喂自纠）与 `output_validation`（局部熔断停用该工具），以及连续两步无新证据提前退出保护；
   * `ANSWER` 节点执行确定性规则校验：剔除非法引用，并在存在 `timeout`/`failed`/`budget_exhausted` 或信息缺口时强制保证 `uncertainties` 非空。
4. **轻量级环境解耦（`deps.py` + `env_factory.py`）**
   * 通过 `RuntimeDeps`（LangGraph `context_schema`）注入检索、只读业务工具、工单存储与可控 `FaultInjector`，核心编排零侵入，支持全离线测试、60 条任务基准评测与真实 PostgreSQL 模式无缝切换。

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. 运行离线自动化测试（无需 API Key 与 PostgreSQL）

```bash
pytest -v
```

覆盖测试项：
* `tests/test_core_flows.py`：端到端双 Agent + `ASK` 追问恢复、瞬时超时 1 次重试及跨循环计数拦截、`output_validation` 局部熔断 + `ANSWER` 强制补齐 `uncertainties`、2 步无新证据提前退出、以及 `OP` 租户沙箱拦截。
* `tests/test_ticket_idempotency.py`：工单审核修改（`v1 -> edit -> v2 -> approve`）与过期版本号保护、同键串行 10 次与并发 10 次防重建、同键异内容 `conflict`、以及 `db_commit_unknown` 后入口直达 `execute_submission` 找回原工单。

### 3. 运行 60 条任务基准评测（多 Agent 协同）

```bash
python scripts/run_eval.py --out-json eval_report.json --out-md eval_report.md
```

* 默认使用 `--mode simulation` 离线确定性评测（`max_rounds=1, max_tool_attempts=8, max_agent_steps_per_task=5`），输出多 Agent 系统分类别报表：
  * `multi_agent`：**50/60 (83.3%)**，其中跨领域综合 **16/20**
* 若需连接真实大模型与 BGE-M3：先配置 `.env`，再运行 `python scripts/run_eval.py --mode live --use-real-embedder`。

### 4. 启动交互式工作台（FastAPI + Streamlit）

**方式 A：零配置离线仿真演示模式（默认）**

终端 1 启动 FastAPI：
```bash
uvicorn supportflow.api.server:app --reload --port 8000
```

终端 2 启动 Streamlit：
```bash
streamlit run src/supportflow/api/ui.py
```

**推荐演示剧本：**
1. **追问恢复**：左侧选择“场景 1”，确认未锁定账户后提交问题 -> 流程暂停在 `ASK` -> 补充 `acc-001` 恢复执行并查看右侧双 Agent 证据卡片。
2. **工单落库超时 `UNKNOWN` 恢复**：左侧勾选“工单落库后响应超时（测 UNKNOWN 恢复）” -> 选择“场景 2”提交 -> 在 `REVIEW` 页面可先试“修改意见”生成 `v2` 草稿，再点“批准并创建工单” -> 界面进入 `unknown` 状态并展示保留的幂等键 -> 点击 **“核对或重试（直达 execute_submission 节点）”** -> 返回原工单 ID，且右侧“全局工单存储”中始终只有 1 张工单。

**方式 B：连接真实 PostgreSQL（pgvector + Checkpointer）与真实 LLM**

1. 复制 `.env.example` 为 `.env`，填入 `SUPPORTFLOW_LLM_API_KEY`、`SUPPORTFLOW_DATABASE_URL`，并将 `SUPPORTFLOW_RUNTIME_MODE=live`。
2. 确保 PostgreSQL 已安装 `pgvector` 扩展并启动，初始化数据库表与示例向量知识库：
   ```bash
   python scripts/init_db.py
   ```
   （若暂不希望下载本地 `BAAI/bge-m3` 权重，可先加 `--test-embedder` 参数，并在 `.env` 将 `SUPPORTFLOW_EMBEDDING_DIM` 改为 `128` 联调。）
3. 随后按上述相同命令启动 `uvicorn` 与 `streamlit` 即可。
