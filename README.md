# SupportFlow

一个面向云存储 SaaS 排障场景的多 Agent 技术支持与工单工作台。

在做这个项目的过程中，我主要想解决大模型落地排障场景时的几个实际工程问题：
1. **多 Agent 并发时的状态污染**：如果让子 Agent 直接读写主图消息列表，工具报错和中间检索文本很快就会把上下文撑爆。
2. **工具调用死循环与契约不可靠**：模型参数填错、后端接口超时、或者接口返回格式异常时，不能无休止重试把 Token 烧光。
3. **排障转工单的一致性**：从 AI 生成工单草稿、人工修改审核（Human-in-the-loop），到最终落库，如何防止并发重复点击或网络超时重试产生重复工单。

技术栈：`Python 3.11+` · `LangGraph` · `FastAPI` · `PostgreSQL (pgvector)` · `BGE-M3` · `BM25` · `Streamlit`

---

## 系统架构

### 1. 主图流程（状态流转与工单幂等事务）

```mermaid
flowchart TD
    START([用户提交问题]) --> ROUTE{入口路由 route_entry}
    ROUTE -->|常规诊断| SUP[sup: 分析问题与信息缺口]
    ROUTE -->|unknown 工单重核| EXEC_SUB

    SUP -->|缺少关键信息| ASK[ask: interrupt 暂停追问]
    ASK -->|用户补充后恢复| SUP
    SUP -->|需要调查| PLAN[plan: 生成调查计划与分配预算]
    SUP -->|已有充分证据| DECIDE

    PLAN --> EXEC[exec: 并发执行子 Agent<br/>各自独立上下文 & 限时等待]
    EXEC --> MERGE[merge: 汇总证据、识别部分失败与冲突]
    MERGE --> DECIDE{decide: 决定下一步}

    DECIDE -->|继续调查且有预算| PLAN
    DECIDE -->|需要用户补充| ASK
    DECIDE -->|生成建议或预算耗尽| ANSWER[answer: 输出排查建议<br/>强校验引用与不确定性说明]
    DECIDE -->|进入工单流程| DRAFT

    subgraph TICKET["工单流程 · 按 Checkpointer 边界拆分"]
        DRAFT[draft: 生成工单草稿] --> REVIEW[review: interrupt 等待人工审核]
        REVIEW -->|edit| EDIT[edit: 更新草稿且 version + 1]
        EDIT --> REVIEW
        REVIEW -->|cancel| CANCEL[cancel: 取消建单]
        REVIEW -->|approve| PREP_SUB[prepare_submission: 固化快照与幂等键落盘]
        PREP_SUB --> EXEC_SUB[execute_submission: LOOKUP + 事务写入 + 冲突回滚重查]
    end

    ANSWER --> END([保存状态并结束])
    CANCEL --> END
    EXEC_SUB --> END
```

### 2. 调查执行节点（`exec`）内部流程

子 Agent 的内部循环不拆分为主图节点，而是在 `exec` 节点内通过异步包装器并发调度，执行完毕后一次性返回结构化 `DiagnosisResult`：

```mermaid
flowchart TD
    INPUT[接收本轮 RoundPlan] --> DISPATCH{并发启动任务}
    DISPATCH -->|知识诊断| K[Knowledge Agent · 独立上下文]
    DISPATCH -->|运行诊断| O[Operations Agent · 独立上下文]

    K --> KP{Pydantic 入参校验}
    KP -->|通过| KT[BGE-M3 + pgvector & BM25 + RRF<br/>版本过滤 & 保留技术标识]
    KP -->|失败| KE[记录参数错误并回喂自纠]
    KT --> KV{出参结构校验}
    KV -->|通过| KD[记录文档证据]
    KV -->|失败| K_CB[局部熔断停用该工具]

    O --> OP{Pydantic 入参 + 租户沙箱校验}
    OP -->|通过| OT[调用只读业务工具<br/>权限 / 配额 / 服务状态 / 请求日志]
    OP -->|失败| OE[记录参数错误并回喂自纠]
    OT -->|瞬时超时且未重试| RETRY[复用原参数原地重试 1 次]
    RETRY --> OT
    OT -->|返回数据| OV{出参结构校验}
    OV -->|通过| OD[记录业务证据: normal / business_error / no_data]
    OV -->|失败| O_CB[局部熔断停用该工具]

    KD --> WRAP[run_wrapped 超时与异常包装器]
    KE --> WRAP
    K_CB --> WRAP
    OD --> WRAP
    OE --> WRAP
    O_CB --> WRAP
    WRAP --> COLLECT[按 task_id 收集结果交回 Supervisor]
```

---

## 核心工程设计说明

- **主子两层 State 隔离 (`state.py` / `nodes/investigation.py`)**
  主图 `SupportState` 只保存用户对话、轮次计划、提炼后的证据字典（`Evidence`）和工单状态，由 PostgreSQL Checkpointer 持久化；子 Agent 运行期的消息流、工具重试记录全部收敛在进程内的 `AgentLocalState` 中，避免中间过程污染主图上下文。
- **跨循环逻辑调用计数与分级容错 (`agents/runner.py`)**
  大模型重新进入循环时会生成新的 `tool_call_id`，因此不能直接用它做重试计数。项目中通过 `hash(task_id + tool_name + canonical_json(args))` 生成稳定的 `logical_call_id`：
  - **入参校验失败 (`argument_validation`)**：不扣减工具调用次数预算，将校验错误塞回消息让模型修正参数；
  - **瞬时超时/网络异常 (`TransientToolError`)**：运行诊断工具在预算允许下直接复用原参数原地重试 1 次（无需额外调用 LLM），且跨循环不重置计数；
  - **出参契约损坏 (`output_validation`)**：后端返回格式不符合 Schema 时，模型改入参也无济于事，直接将该工具在当前轮次局部熔断；
  - **防空转退出**：连续 2 步未产出任何新证据时提前退出循环。
- **工单快照落盘与幂等边界 (`nodes/ticket.py` / `persistence/ticket_service.py`)**
  由于 LangGraph 在包含 `interrupt()` 的节点恢复时会从头重跑该节点，且 Checkpointer 在节点边界落盘，因此将工单确认后的流程拆分为 `prepare_submission` 和 `execute_submission` 两个独立节点：
  1. `prepare_submission` 根据 `case_id + draft_id + version` 生成幂等键 `idempotency_key` 与 `payload_hash`，先持久化到 Checkpoint；
  2. `execute_submission` 执行前置 `LOOKUP` 与 PostgreSQL 事务写入，利用 `(account_id, idempotency_key)` 唯一约束拦截并发竞争（冲突时回滚当前事务并重新查询返回已有工单）；
  3. 当遇到“数据库已提交但网络响应超时”（`unknown` 状态）时，保留原快照，后续重试由图入口直接路由到 `execute_submission` 查库恢复。

---

## 目录结构

```text
supportflow/
├── src/supportflow/
│   ├── config.py                # 配置管理 (pydantic-settings)
│   ├── state.py                 # SupportState 与 AgentLocalState 定义
│   ├── schemas.py               # 工具入参/出参 Schema、结构化输出与异常定义
│   ├── deps.py                  # 运行时依赖注入容器 RuntimeDeps
│   ├── context_policy.py        # 角色上下文投影、工具返回截断与数据定界标签
│   ├── graph.py                 # 13 节点 LangGraph 主图编排与路由
│   ├── agents/runner.py         # 通用子 Agent 执行循环
│   ├── nodes/
│   │   ├── investigation.py     # exec 并发节点与 run_wrapped 超时包装器
│   │   ├── supervisor.py        # sup / ask / plan / merge / decide / answer
│   │   └── ticket.py            # draft / review / edit / cancel / prepare / execute
│   ├── tools/
│   │   ├── knowledge.py         # BGE-M3 + pgvector + BM25 + RRF 混合检索
│   │   └── operations.py        # 4 类只读业务诊断工具与故障注入网关
│   ├── persistence/
│   │   ├── db.py                # PostgreSQL 连接池、表结构与 Checkpointer 初始化
│   │   └── ticket_service.py    # 幂等工单事务服务
│   └── api/
│       ├── server.py            # FastAPI 服务端 (含会话级并发锁)
│       └── ui.py                # Streamlit 交互式工作台
├── scripts/
│   ├── init_db.py               # 初始化数据库表与示例知识库分块
│   └── run_eval.py              # 60 条任务基准评测脚本
└── tests/                       # 核心链路与工单幂等并发单元测试
```

---

## 快速开始

### 1. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. 配置与启动服务

**方式一：连接 PostgreSQL 与大模型 API（完整模式）**

1. 复制 `.env.example` 为 `.env`，配置数据库连接串与模型密钥（兼容 OpenAI / DeepSeek 接口）：
   ```ini
   SUPPORTFLOW_RUNTIME_MODE=live
   SUPPORTFLOW_LLM_API_KEY=sk-xxx
   SUPPORTFLOW_LLM_BASE_URL=https://api.deepseek.com/v1
   SUPPORTFLOW_LLM_MODEL=deepseek-chat
   SUPPORTFLOW_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/supportflow
   ```
2. 初始化 PostgreSQL 表结构（需安装 `pgvector` 扩展）并写入示例文档向量：
   ```bash
   python scripts/init_db.py
   ```
3. 启动后端与前端工作台：
   ```bash
   # 终端 1：启动 FastAPI
   uvicorn supportflow.api.server:app --reload --port 8000

   # 终端 2：启动 Streamlit
   streamlit run src/supportflow/api/ui.py
   ```

**方式二：本地免配置快速体验（内置 Mock 模式）**

如果本地暂未启动 PostgreSQL，项目内置了内存存储与确定性仿真环境（默认 `SUPPORTFLOW_RUNTIME_MODE=simulation`），直接运行上述 `uvicorn` 和 `streamlit` 命令即可体验完整的追问中断（`ASK`）、双 Agent 证据溯源、工单草稿修改（`REVIEW/EDIT`）及故障注入演示。

---

## 测试与基准评测

### 1. 运行单元测试

测试集覆盖了双 Agent 协作与追问恢复、工具瞬时超时重试与跨循环上限拦截、出参契约熔断、`OP` 租户沙箱越权拦截、以及工单同键串行 10 次 / 并发 10 次提交与 `unknown` 状态恢复：

```bash
pytest -v
```

### 2. 运行 60 条基准评测

评测集包含 20 条知识检索任务、20 条运行诊断任务与 20 条跨领域综合任务（含无答案版本与工具异常边界用例），从引用真实性、证据类型覆盖度、关键词命中及异常诚实度四个维度判定任务是否完成：

```bash
python scripts/run_eval.py --out-json eval_report.json --out-md eval_report.md
```

| 评测维度 | 任务数量 | 完成数 / 总数 | 完成率 |
| :--- | :---: | :---: | :---: |
| **总体任务** | 60 | **50 / 60** | **83.3%** |
| 知识检索类 (`knowledge`) | 20 | 17 / 20 | 85.0% |
| 运行诊断类 (`operations`) | 20 | 17 / 20 | 85.0% |
| 跨领域综合类 (`cross_domain`) | 20 | 16 / 20 | 80.0% |
