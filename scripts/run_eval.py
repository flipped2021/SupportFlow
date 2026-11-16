from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from supportflow.config import Settings, get_settings  # noqa: E402
from supportflow.env_factory import (  # noqa: E402
    create_chat_model,
    create_embedder,
    create_simulation_deps,
)
from supportflow.graph import build_support_graph  # noqa: E402
from supportflow.persistence.ticket_service import InMemoryTicketStore  # noqa: E402
from supportflow.schemas import FaultInjectorConfig, FaultRule  # noqa: E402
from supportflow.state import BudgetPolicy, SupportState, create_initial_state  # noqa: E402
from supportflow.tools import FaultInjector  # noqa: E402
from supportflow.tools.knowledge import (  # noqa: E402
    DEFAULT_SAAS_DOCUMENTS,
    InMemoryHybridKnowledgeRetriever,
    KnowledgeChunkRecord,
)
from supportflow.tools.operations import (  # noqa: E402
    SimulationOperationsGateway,
    default_saas_accounts,
    default_service_statuses,
)

TaskCategory = Literal["knowledge", "operations", "cross_domain"]


@dataclass(frozen=True, slots=True)
class EvalTaskCase:
    task_id: str
    category: TaskCategory
    question: str
    account_id: str | None = None
    product_version: str | None = None
    request_ids: tuple[str, ...] = ()
    required_evidence_kinds: tuple[Literal["document", "business"], ...] = ("document",)
    expected_keywords: tuple[str, ...] = ()
    fault_rules: tuple[FaultRule, ...] = ()
    expect_solvable: bool = True


@dataclass(slots=True)
class EvalCaseOutcome:
    task_id: str
    category: TaskCategory
    passed: bool
    reason: str
    tool_attempts: int
    agent_steps: int
    elapsed_ms: int
    cited_evidence_count: int
    cited_kinds: list[str] = field(default_factory=list)
    task_statuses: list[str] = field(default_factory=list)


def _build_eval_corpus() -> list[KnowledgeChunkRecord]:
    """在默认 SaaS 文档基础上补充评测覆盖的扩展知识条目。"""
    extra = [
        KnowledgeChunkRecord(
            document_id="webhook-retry-spec",
            document_version="v2.3",
            chunk_id="chunk-01",
            title="v2.3 上传回调 Webhook 重试与签名规范",
            text=(
                "文件上传完成后触发的 Webhook 回调使用 X-SF-Signature-V2 请求头，"
                "失败后采用指数退避最多重试 5 次，配置项 WEBHOOK_MAX_RETRIES=5，"
                "超时阈值 WEBHOOK_TIMEOUT_MS=3000。"
            ),
        ),
        KnowledgeChunkRecord(
            document_id="sts-token-policy",
            document_version="v2.3",
            chunk_id="chunk-01",
            title="v2.3 临时 STS 凭证有效期与刷新说明",
            text=(
                "临时 STS 上传凭证默认有效期 STS_TTL_SECONDS=3600 秒，"
                "过期后网关返回 HTTP 401（错误码 E_AUTH_TOKEN_EXPIRED）。"
                "请勿与权限不足导致的 HTTP 403（E_UPLOAD_403）混淆。"
            ),
        ),
        KnowledgeChunkRecord(
            document_id="rate-limit-guide",
            document_version="v2.2",
            chunk_id="chunk-01",
            title="v2.2 API 调用频率限流说明",
            text=(
                "v2.2 版本单账户默认每日 API 调用配额由 daily_api_limit 控制，"
                "触发限流后返回 HTTP 429 与错误码 E_RATE_LIMIT_EXCEEDED。"
            ),
        ),
    ]
    return [*DEFAULT_SAAS_DOCUMENTS, *extra]


def _build_eval_accounts() -> dict[str, dict[str, Any]]:
    accounts = default_saas_accounts()
    accounts.update(
        {
            "acc-005": {
                "tier": "pro",
                "status": "active",
                "permissions": ["storage:read", "storage:write"],
                "token_scopes": ["storage:read"],
                "can_upload": False,
                "storage_used_bytes": 1_500_000_000,
                "storage_limit_bytes": 50_000_000_000,
                "daily_api_calls": 900,
                "daily_api_limit": 100_000,
                "logs": [
                    {
                        "request_id": "req-505",
                        "timestamp": "2026-11-16T09:50:00Z",
                        "endpoint": "POST /api/v2/files/upload",
                        "http_status": 403,
                        "error_code": "E_UPLOAD_403",
                        "reason": "missing storage:write scope",
                        "product_version": "v2.3",
                    }
                ],
            },
            "acc-006": {
                "tier": "starter",
                "status": "active",
                "permissions": ["storage:read", "storage:write"],
                "token_scopes": ["storage:read", "storage:write"],
                "can_upload": True,
                "storage_used_bytes": 2_000_000_000,
                "storage_limit_bytes": 10_737_418_240,
                "daily_api_calls": 10_000,
                "daily_api_limit": 10_000,
                "logs": [
                    {
                        "request_id": "req-606",
                        "timestamp": "2026-11-16T10:15:00Z",
                        "endpoint": "POST /api/v2/files/upload",
                        "http_status": 429,
                        "error_code": "E_RATE_LIMIT_EXCEEDED",
                        "reason": "daily_api_calls reached daily_api_limit",
                        "product_version": "v2.3",
                    }
                ],
            },
            "acc-007": {
                "tier": "enterprise",
                "status": "active",
                "permissions": ["storage:read", "storage:write"],
                "token_scopes": ["storage:read", "storage:write"],
                "can_upload": True,
                "storage_used_bytes": 5_000_000_000,
                "storage_limit_bytes": 500_000_000_000,
                "daily_api_calls": 1200,
                "daily_api_limit": 500_000,
                "logs": [
                    {
                        "request_id": "req-707",
                        "timestamp": "2026-11-16T11:30:00Z",
                        "endpoint": "POST /api/v2/files/multipart/init",
                        "http_status": 503,
                        "error_code": "E_GATEWAY_REGION_DEGRADED",
                        "reason": "cn-east-1 gateway degraded (INC-2026-1104)",
                        "product_version": "v2.3",
                    }
                ],
            },
            "acc-008": {
                "tier": "pro",
                "status": "restricted",
                "permissions": ["storage:read"],
                "token_scopes": ["storage:read"],
                "can_upload": False,
                "storage_used_bytes": 500_000_000,
                "storage_limit_bytes": 50_000_000_000,
                "daily_api_calls": 42,
                "daily_api_limit": 100_000,
                "logs": [
                    {
                        "request_id": "req-808",
                        "timestamp": "2026-11-16T08:05:00Z",
                        "endpoint": "POST /api/v2/files/upload",
                        "http_status": 403,
                        "error_code": "E_ACCOUNT_RESTRICTED",
                        "reason": "account status restricted",
                        "product_version": "v2.3",
                    }
                ],
            },
        }
    )
    return accounts


def build_benchmark_tasks() -> list[EvalTaskCase]:
    """构建完整的 60 条评测任务集：20 知识检索 + 20 运行诊断 + 20 跨领域综合。"""
    tasks: list[EvalTaskCase] = []

    # ------------------------------------------------------------------
    # 一、20 条知识检索类任务（K-01 ~ K-20：17 条可解 + 3 条无文档边界）
    # ------------------------------------------------------------------
    knowledge_specs = [
        ("K-01", "v2.3", "请问 v2.3 文件上传接口需要什么权限范围（文档规范是什么）？", ("storage:write",)),
        ("K-02", "v2.3", "请说明 v2.3 上传返回 E_UPLOAD_403 的文档规则与校验开关。", ("STRICT_SCOPE_CHECK_V23",)),
        ("K-03", "v2.3", "从 v2.2 升级至 v2.3 后，上传权限兼容开关有哪些变化？文档怎么说明？", ("IMPLICIT_WRITE_LEGACY",)),
        ("K-04", "v2.2", "v2.2 文件上传接口权限说明中对默认读写令牌是怎么规定的？", ("IMPLICIT_WRITE_LEGACY",)),
        ("K-05", "v2.3", "v2.3 文档对存储空间配额超限和 API 频率限制的错误码规范是什么？", ("E_QUOTA_STORAGE_FULL",)),
        ("K-06", "v2.3", "v2.3 单分片大小上限配置项 MAX_PART_SIZE_MB 的文档规定是多少？", ("MAX_PART_SIZE_MB",)),
        ("K-07", "v2.3", "文档中 E_ACCOUNT_RESTRICTED 对应什么账户状态？", ("suspended",)),
        ("K-08", "v2.3", "文档中 E_GATEWAY_REGION_DEGRADED 错误码通常伴随什么现象？", ("503",)),
        ("K-09", "v2.3", "大文件分片上传初始化接口与 MULTIPART_UPLOAD_TIMEOUT_MS 文档规范是什么？", ("30000",)),
        ("K-10", "v2.3", "v2.3 上传回调 Webhook 的签名头和重试配置项文档规范是什么？", ("WEBHOOK_MAX_RETRIES",)),
        ("K-11", "v2.3", "v2.3 临时 STS 凭证有效期 STS_TTL_SECONDS 与过期错误码文档怎么规定？", ("E_AUTH_TOKEN_EXPIRED",)),
        ("K-12", "v2.2", "v2.2 文档中 API 调用频率超限会返回什么错误码？", ("E_RATE_LIMIT_EXCEEDED",)),
        ("K-13", "v2.3", "为什么 v2.3 文档强调不能混淆 E_AUTH_TOKEN_EXPIRED 和 E_UPLOAD_403？", ("401",)),
        ("K-14", "v2.3", "当区域网关健康状态为 degraded 时，文档建议如何处置？", ("degraded",)),
        ("K-15", "v2.3", "v2.3 文档中分片上传超过多少 MB 需要调用 multipart/init？", ("64",)),
        ("K-16", "v2.3", "v2.3 上传回调 Webhook 超时配置项 WEBHOOK_TIMEOUT_MS 文档规范值是多少？", ("3000",)),
        ("K-17", "v2.2", "v2.2 文档中遇到 E_ACCOUNT_SUSPENDED 应先核实什么规范？", ("suspended",)),
    ]
    for tid, ver, q, kws in knowledge_specs:
        tasks.append(
            EvalTaskCase(
                task_id=tid,
                category="knowledge",
                question=q,
                account_id=None,
                product_version=ver,
                required_evidence_kinds=("document",),
                expected_keywords=kws,
                expect_solvable=True,
            )
        )

    # 3 条未覆盖版本文档边界（验证诚实退出而非编造文档依据）
    for tid, ver, q in [
        ("K-18", "v1.9", "请问 v1.9 老版本中插件热加载配置文档规范是什么？"),
        ("K-19", "v2.4", "请问未发布的 v2.4 文档中灰度流量权重参数规范是什么？"),
        ("K-20", "v1.0", "请问 v1.0 历史归档文档中对象加密密钥轮转周期规范是什么？"),
    ]:
        tasks.append(
            EvalTaskCase(
                task_id=tid,
                category="knowledge",
                question=q,
                account_id=None,
                product_version=ver,
                required_evidence_kinds=("document",),
                expected_keywords=(),
                expect_solvable=False,
            )
        )

    # ------------------------------------------------------------------
    # 二、20 条运行诊断类任务（O-01 ~ O-20：17 条可解 + 3 条不可恢复故障）
    # ------------------------------------------------------------------
    ops_specs = [
        ("O-01", "acc-001", "v2.3", (), "请核查账户 acc-001 当前的上传权限与令牌作用域。", ("storage:read",)),
        ("O-02", "acc-001", "v2.3", ("req-123",), "请核查账户 acc-001 权限及请求 req-123 的日志详情（403）。", ("E_UPLOAD_403",)),
        ("O-03", "acc-002", "v2.3", (), "请核查账户 acc-002 的存储配额使用状态（429）。", ("quota_exceeded=True",)),
        ("O-04", "acc-002", "v2.3", ("req-202",), "请核查账户 acc-002 配额及请求 req-202 日志记录。", ("E_QUOTA_STORAGE_FULL",)),
        ("O-05", "acc-003", "v2.3", (), "请检查 cn-east-1 区域网关服务健康状态（503，账户 acc-003）。", ("INC-2026-1104",)),
        ("O-06", "acc-003", "v2.3", (), "请核查 cn-south-1 区域网关服务健康状态（账户 acc-003）。", ("healthy",)),
        ("O-07", "acc-004", "v2.3", (), "请核查账户 acc-004 当前账户状态与上传权限。", ("suspended",)),
        ("O-08", "acc-004", "v2.3", ("req-404",), "请核查账户 acc-004 权限及请求 req-404 日志（403）。", ("E_ACCOUNT_RESTRICTED",)),
        ("O-09", "acc-005", "v2.3", (), "请核查账户 acc-005 的权限列表与令牌作用域（403）。", ("storage:read",)),
        ("O-10", "acc-005", "v2.3", ("req-505",), "请核查账户 acc-005 权限及请求 req-505 日志详情。", ("E_UPLOAD_403",)),
        ("O-11", "acc-006", "v2.3", (), "请核查账户 acc-006 的每日 API 配额消耗状态（429）。", ("quota_exceeded=True",)),
        ("O-12", "acc-006", "v2.3", ("req-606",), "请核查账户 acc-006 配额及请求 req-606 日志记录。", ("E_RATE_LIMIT_EXCEEDED",)),
        ("O-13", "acc-007", "v2.3", (), "请检查 cn-east-1 区域服务状态与事件编号（503，账户 acc-007）。", ("degraded",)),
        ("O-14", "acc-007", "v2.3", ("req-707",), "请核查账户 acc-007 在请求 req-707 上的服务端日志（503）。", ("E_GATEWAY_REGION_DEGRADED",)),
        ("O-15", "acc-008", "v2.3", (), "请核查账户 acc-008 当前状态是否受限及上传权限。", ("restricted",)),
        ("O-16", "acc-008", "v2.3", ("req-808",), "请核查账户 acc-008 权限及请求 req-808 失败日志。", ("E_ACCOUNT_RESTRICTED",)),
        ("O-17", "acc-001", "v2.3", ("req-120",), "请核查账户 acc-001 请求 req-120 的日志返回状态。", ("200",)),
    ]
    for tid, acc, ver, reqs, q, kws in ops_specs:
        tasks.append(
            EvalTaskCase(
                task_id=tid,
                category="operations",
                question=q,
                account_id=acc,
                product_version=ver,
                request_ids=reqs,
                required_evidence_kinds=("business",),
                expected_keywords=kws,
                expect_solvable=True,
            )
        )

    # 3 条运行态不可恢复边界任务
    tasks.extend(
        [
            EvalTaskCase(
                task_id="O-18",
                category="operations",
                question="请核查不存在的账户 acc-missing 的上传权限与配额。",
                account_id="acc-missing",
                product_version="v2.3",
                required_evidence_kinds=("business",),
                expect_solvable=False,
            ),
            EvalTaskCase(
                task_id="O-19",
                category="operations",
                question="请核查账户 acc-001 当前的上传权限状态（403）。",
                account_id="acc-001",
                product_version="v2.3",
                required_evidence_kinds=("business",),
                fault_rules=(
                    FaultRule(
                        target="get_account_permissions",
                        fault_type="invalid_output_schema",
                        trigger_on_attempt=None,
                    ),
                ),
                expect_solvable=False,
            ),
            EvalTaskCase(
                task_id="O-20",
                category="operations",
                question="请核查账户 acc-002 的存储配额是否已满（429）。",
                account_id="acc-002",
                product_version="v2.3",
                required_evidence_kinds=("business",),
                fault_rules=(
                    FaultRule(
                        target="get_account_quota",
                        fault_type="timeout",
                        trigger_on_attempt=None,
                    ),
                ),
                expect_solvable=False,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # 三、20 条跨领域综合类任务（C-01 ~ C-20）
    #   - C-01 ~ C-12: 常规跨领域任务
    #   - C-13 ~ C-16: 复杂多步跨领域任务（需知识检索与运行诊断并行协作、各自独立预算完成）
    #   - C-17 ~ C-20: 跨领域不可解/故障边界（验证诚实退出而非蒙混通过）
    # ------------------------------------------------------------------
    cross_simple_specs = [
        ("C-01", "acc-001", "v2.3", ("req-123",), "账户 acc-001 在 v2.3 上传文件返回 403（req-123），请结合文档规范与运行状态排查原因。", ("storage:write",)),
        ("C-02", "acc-001", "v2.3", (), "账户 acc-001 在 v2.3 遇到 E_UPLOAD_403 报错，请结合产品权限文档与账户权限定位问题。", ("storage:write",)),
        ("C-03", "acc-002", "v2.3", ("req-202",), "账户 acc-002 在 v2.3 触发配额限制 429（请求 req-202），请结合文档与日志配额核查。", ("quota_exceeded=True",)),
        ("C-04", "acc-002", "v2.3", (), "账户 acc-002 在 v2.3 出现 E_QUOTA_STORAGE_FULL，请结合配额文档与账户数据诊断。", ("storage_limit_bytes",)),
        ("C-05", "acc-003", "v2.3", ("req-303",), "账户 acc-003 在 cn-east-1 网关报 503（req-303），请结合错误码文档与服务健康度分析。", ("INC-2026-1104",)),
        ("C-06", "acc-003", "v2.3", (), "v2.3 下 cn-east-1 出现 E_GATEWAY_REGION_DEGRADED，请结合文档建议与服务状态（账户 acc-003）排查。", ("degraded",)),
        ("C-07", "acc-004", "v2.3", ("req-404",), "账户 acc-004 在 v2.3 上传报 403（req-404），请结合错误码文档与账户状态排查。", ("suspended",)),
        ("C-08", "acc-005", "v2.3", ("req-505",), "账户 acc-005 在 v2.3 上传返回 403（req-505），请结合权限规范文档与账户日志排查。", ("storage:write",)),
        ("C-09", "acc-006", "v2.3", ("req-606",), "账户 acc-006 在 v2.3 遭遇 429 频率限流（req-606），请结合配额文档与请求日志诊断。", ("E_RATE_LIMIT_EXCEEDED",)),
        ("C-10", "acc-007", "v2.3", ("req-707",), "账户 acc-007 在 cn-east-1 分片初始化返回 503（req-707），请结合文档与服务状态排查。", ("degraded",)),
        ("C-11", "acc-008", "v2.3", ("req-808",), "账户 acc-008 在 v2.3 上传返回 403（req-808），请结合文档说明与账户状态排查。", ("restricted",)),
        ("C-12", "acc-005", "v2.3", (), "账户 acc-005 在 v2.3 遇到上传 403，请结合 STRICT_SCOPE_CHECK_V23 文档与账户令牌权限分析。", ("storage:write",)),
    ]
    for tid, acc, ver, reqs, q, kws in cross_simple_specs:
        tasks.append(
            EvalTaskCase(
                task_id=tid,
                category="cross_domain",
                question=q,
                account_id=acc,
                product_version=ver,
                request_ids=reqs,
                required_evidence_kinds=("document", "business"),
                expected_keywords=kws,
                expect_solvable=True,
            )
        )

    # C-13 ~ C-16：复杂多跳跨领域任务（需补读升级分块 + 多项业务工具钻取）
    cross_complex_specs = [
        (
            "C-13",
            "acc-002",
            "v2.3",
            ("req-202",),
            "账户 acc-002 从 v2.2 升级至 v2.3 后上传被拒（req-202），请结合升级变更文档核查权限、配额与请求日志。",
            ("E_QUOTA_STORAGE_FULL",),
        ),
        (
            "C-14",
            "acc-003",
            "v2.3",
            ("req-303",),
            "账户 acc-003 从 v2.2 升级至 v2.3 后在 cn-east-1 分片上传返回 503（req-303），请结合升级文档核查权限、配额、日志与服务状态。",
            ("INC-2026-1104",),
        ),
        (
            "C-15",
            "acc-001",
            "v2.3",
            ("req-123",),
            "账户 acc-001 从 v2.2 升级到 v2.3 后上传返回 403（req-123），请结合升级说明核查权限、配额与请求日志。",
            ("storage:write",),
        ),
        (
            "C-16",
            "acc-004",
            "v2.3",
            ("req-404",),
            "账户 acc-004 从 v2.2 升级到 v2.3 后上传报 403（req-404），请结合升级说明核查权限、配额与请求日志。",
            ("suspended",),
        ),
    ]
    for tid, acc, ver, reqs, q, kws in cross_complex_specs:
        tasks.append(
            EvalTaskCase(
                task_id=tid,
                category="cross_domain",
                question=q,
                account_id=acc,
                product_version=ver,
                request_ids=reqs,
                required_evidence_kinds=("document", "business"),
                expected_keywords=kws,
                expect_solvable=True,
            )
        )

    # C-17 ~ C-20：跨领域不可解/故障边界（双方均不通过）
    tasks.extend(
        [
            EvalTaskCase(
                task_id="C-17",
                category="cross_domain",
                question="账户 acc-missing 在 v2.3 上传 403，请结合文档与账户权限排查。",
                account_id="acc-missing",
                product_version="v2.3",
                required_evidence_kinds=("document", "business"),
                expect_solvable=False,
            ),
            EvalTaskCase(
                task_id="C-18",
                category="cross_domain",
                question="账户 acc-001 在未发布版本 v2.4 上传报错，请结合 v2.4 文档与账户权限排查。",
                account_id="acc-001",
                product_version="v2.4",
                required_evidence_kinds=("document", "business"),
                expect_solvable=False,
            ),
            EvalTaskCase(
                task_id="C-19",
                category="cross_domain",
                question="账户 acc-002 在 v2.3 报配额限制，请结合配额文档与账户配额诊断。",
                account_id="acc-002",
                product_version="v2.3",
                required_evidence_kinds=("document", "business"),
                fault_rules=(
                    FaultRule(
                        target="get_account_quota",
                        fault_type="invalid_output_schema",
                        trigger_on_attempt=None,
                    ),
                ),
                expect_solvable=False,
            ),
            EvalTaskCase(
                task_id="C-20",
                category="cross_domain",
                question="账户 acc-003 在 v2.3 报 503 服务不可用，请结合错误码文档与服务健康状态排查。",
                account_id="acc-003",
                product_version="v2.3",
                required_evidence_kinds=("document", "business"),
                fault_rules=(
                    FaultRule(
                        target="get_service_status",
                        fault_type="timeout",
                        trigger_on_attempt=None,
                    ),
                ),
                expect_solvable=False,
            ),
        ]
    )

    assert len(tasks) == 60, f"预期 60 条评测任务，实际为 {len(tasks)}"
    return tasks


def evaluate_task_outcome(
    case: EvalTaskCase,
    final_state: SupportState,
    *,
    elapsed_ms: int,
) -> EvalCaseOutcome:
    """严格判定单条任务的完成情况（拒绝幻觉引用、缺证据蒙混与未声明的预算截断）。"""
    answer = final_state.get("answer")
    evidence_pool = final_state.get("evidence") or {}

    task_statuses: list[str] = []
    total_attempts = 0
    total_steps = 0
    for rdata in (final_state.get("rounds") or {}).values():
        for res in rdata.get("results") or []:
            task_statuses.append(str(res["status"]))
            total_attempts += int(res["usage"]["tool_attempts"])
            total_steps += int(res["usage"]["agent_steps"])

    if answer is None or not answer["content"].strip():
        return EvalCaseOutcome(
            task_id=case.task_id,
            category=case.category,
            passed=False,
            reason="未生成有效回答",
            tool_attempts=total_attempts,
            agent_steps=total_steps,
            elapsed_ms=elapsed_ms,
            cited_evidence_count=0,
            task_statuses=task_statuses,
        )

    citations = list(answer["citation_evidence_ids"])
    if not citations:
        return EvalCaseOutcome(
            task_id=case.task_id,
            category=case.category,
            passed=False,
            reason="回答未引用任何有效证据 ID",
            tool_attempts=total_attempts,
            agent_steps=total_steps,
            elapsed_ms=elapsed_ms,
            cited_evidence_count=0,
            task_statuses=task_statuses,
        )

    # 1) 所有 citation_evidence_ids 必须真实存在于 State 证据池
    for eid in citations:
        if eid not in evidence_pool:
            return EvalCaseOutcome(
                task_id=case.task_id,
                category=case.category,
                mode=mode,
                passed=False,
                reason=f"存在非法引用 ID: {eid}",
                tool_attempts=total_attempts,
                agent_steps=total_steps,
                elapsed_ms=elapsed_ms,
                cited_evidence_count=len(citations),
                task_statuses=task_statuses,
            )

    cited_kinds = sorted({evidence_pool[eid]["source"]["kind"] for eid in citations})

    # 2) 覆盖题型要求的必要证据类型（如 cross_domain 必须同时覆盖 document 与 business）
    missing_kinds = [k for k in case.required_evidence_kinds if k not in cited_kinds]
    if missing_kinds:
        return EvalCaseOutcome(
            task_id=case.task_id,
            category=case.category,
            passed=False,
            reason=f"缺少题型要求证据类型: {missing_kinds}",
            tool_attempts=total_attempts,
            agent_steps=total_steps,
            elapsed_ms=elapsed_ms,
            cited_evidence_count=len(citations),
            cited_kinds=cited_kinds,
            task_statuses=task_statuses,
        )

    # 3) 拒绝静默失败与预算截断未完成：若有任务非 completed，不可算作完整解决
    if any(st != "completed" for st in task_statuses):
        return EvalCaseOutcome(
            task_id=case.task_id,
            category=case.category,
            passed=False,
            reason=f"存在未正常完成的调查子任务: {task_statuses}",
            tool_attempts=total_attempts,
            agent_steps=total_steps,
            elapsed_ms=elapsed_ms,
            cited_evidence_count=len(citations),
            cited_kinds=cited_kinds,
            task_statuses=task_statuses,
        )

    # 4) 核验核心定位关键词在引用证据或回答中出现
    evidence_blob = "\n".join(evidence_pool[eid]["content"] for eid in citations) + "\n" + answer["content"]
    if case.expected_keywords and not any(kw in evidence_blob for kw in case.expected_keywords):
        return EvalCaseOutcome(
            task_id=case.task_id,
            category=case.category,
            passed=False,
            reason=f"未命中关键诊断依据: {case.expected_keywords}",
            tool_attempts=total_attempts,
            agent_steps=total_steps,
            elapsed_ms=elapsed_ms,
            cited_evidence_count=len(citations),
            cited_kinds=cited_kinds,
            task_statuses=task_statuses,
        )

    return EvalCaseOutcome(
        task_id=case.task_id,
        category=case.category,
        passed=True,
        reason="通过全部证据覆盖与完成标准校验",
        tool_attempts=total_attempts,
        agent_steps=total_steps,
        elapsed_ms=elapsed_ms,
        cited_evidence_count=len(citations),
        cited_kinds=cited_kinds,
        task_statuses=task_statuses,
    )


async def run_single_case(
    case: EvalTaskCase,
    *,
    settings: Settings,
    budget: BudgetPolicy,
    corpus: list[KnowledgeChunkRecord],
    accounts: dict[str, dict[str, Any]],
    use_simulation_llm: bool,
    use_test_embedder: bool,
) -> EvalCaseOutcome:
    fault_cfg = FaultInjectorConfig(rules=list(case.fault_rules))
    injector = FaultInjector(fault_cfg)
    embedder = create_embedder(settings, use_test_embedder=use_test_embedder)
    retriever = InMemoryHybridKnowledgeRetriever(
        embedder=embedder,
        chunks=corpus,
        rrf_k=settings.rrf_k,
        fault_injector=injector,
    )
    gateway = SimulationOperationsGateway(
        accounts=accounts,
        service_statuses=default_service_statuses(),
        fault_injector=injector,
    )
    store = InMemoryTicketStore(fault_injector=injector)
    llm = create_chat_model(settings, use_simulation_llm=use_simulation_llm)

    deps = create_simulation_deps(
        settings=settings,
        fault_config=fault_cfg,
        verified_account_id=case.account_id,
        llm=llm,
        use_test_embedder=use_test_embedder,
        ticket_store=store,
        knowledge_retriever=retriever,
        operations_gateway=gateway,
    )

    graph = build_support_graph(checkpointer=MemorySaver())
    initial = create_initial_state(
        case.question,
        budget,
        case_id=f"eval-{case.task_id}",
        account_id=case.account_id,
        product_version=case.product_version,
        request_ids=list(case.request_ids),
    )

    t0 = time.monotonic()
    final_state = await graph.ainvoke(
        initial,
        config={"configurable": {"thread_id": f"eval:{case.task_id}"}},
        context=deps,
    )
    elapsed_ms = max(1, int((time.monotonic() - t0) * 1000))
    return evaluate_task_outcome(case, final_state, elapsed_ms=elapsed_ms)


def summarize_results(outcomes: list[EvalCaseOutcome]) -> dict[str, Any]:
    def _stats(items: list[EvalCaseOutcome]) -> dict[str, Any]:
        total = len(items)
        passed = sum(1 for x in items if x.passed)
        by_cat: dict[str, dict[str, Any]] = {}
        for cat in ("knowledge", "operations", "cross_domain"):
            sub = [x for x in items if x.category == cat]
            c_pass = sum(1 for x in sub if x.passed)
            by_cat[cat] = {
                "passed": c_pass,
                "total": len(sub),
                "rate": round((c_pass / len(sub)) * 100, 1) if sub else 0.0,
            }
        avg_attempts = round(sum(x.tool_attempts for x in items) / total, 2) if total else 0.0
        avg_ms = round(sum(x.elapsed_ms for x in items) / total, 1) if total else 0.0
        return {
            "passed": passed,
            "total": total,
            "rate": round((passed / total) * 100, 1) if total else 0.0,
            "by_category": by_cat,
            "avg_tool_attempts": avg_attempts,
            "avg_elapsed_ms": avg_ms,
        }

    return {
        "multi_agent": _stats(outcomes),
        "details": [asdict(x) for x in outcomes],
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    m = summary["multi_agent"]
    lines = [
        "# SupportFlow 60 条任务多 Agent 基准评测报告",
        "",
        "| 评测维度 | 多 Agent 协同 (`multi_agent`) |",
        "| :--- | :---: |",
        f"| **总体完成率** | **{m['passed']}/{m['total']} ({m['rate']}%)** |",
    ]
    cat_labels = {
        "knowledge": "知识检索类 (20条)",
        "operations": "运行诊断类 (20条)",
        "cross_domain": "跨领域综合类 (20条)",
    }
    for cat, label in cat_labels.items():
        mc = m["by_category"][cat]
        lines.append(f"| {label} | {mc['passed']}/{mc['total']} ({mc['rate']}%) |")
    lines.extend(
        [
            f"| 平均工具调用次数 | {m['avg_tool_attempts']} |",
            f"| 平均端到端耗时 (ms) | {m['avg_elapsed_ms']} |",
        ]
    )
    return "\n".join(lines)


async def run_benchmark(
    *,
    use_simulation_llm: bool,
    use_test_embedder: bool,
    out_json: Path | None,
    out_md: Path | None,
) -> dict[str, Any]:
    base_settings = get_settings()
    eval_settings = base_settings.model_copy(
        update={
            "default_max_rounds": 1,
            "default_max_tool_attempts": 8,
            "default_max_investigation_ms": 30000,
            "default_max_agent_steps_per_task": 5,
            "default_round_timeout_ms": 15000,
            "default_task_timeout_ms": 12000,
            "default_tool_timeout_ms": 3000,
        }
    )
    # 两组实验严格共享完全一致的全局 BudgetPolicy
    shared_budget = eval_settings.to_budget_policy()
    cases = build_benchmark_tasks()
    corpus = _build_eval_corpus()
    accounts = _build_eval_accounts()

    outcomes: list[EvalCaseOutcome] = []

    for case in cases:
        outcomes.append(
            await run_single_case(
                case,
                settings=eval_settings,
                budget=shared_budget,
                corpus=corpus,
                accounts=accounts,
                use_simulation_llm=use_simulation_llm,
                use_test_embedder=use_test_embedder,
            )
        )

    summary = summarize_results(outcomes)
    md_report = render_markdown_report(summary)
    print("\n" + md_report + "\n")

    if out_json is not None:
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已导出 JSON 评测明细至: {out_json}")
    if out_md is not None:
        out_md.write_text(md_report, encoding="utf-8")
        print(f"已导出 Markdown 评测报告至: {out_md}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportFlow 60 条任务多 Agent 基准评测脚本")
    parser.add_argument(
        "--mode",
        choices=["simulation", "live"],
        default="simulation",
        help="simulation 使用离线规则仿真模型（默认），live 调用 .env 中配置的真实 LLM",
    )
    parser.add_argument(
        "--use-real-embedder",
        action="store_true",
        help="强制使用配置的真实 BGE-M3 模型（默认离线评测采用确定性轻量向量器以加快运行）",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="导出 JSON 报告路径")
    parser.add_argument("--out-md", type=Path, default=None, help="导出 Markdown 报告路径")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            use_simulation_llm=(args.mode == "simulation"),
            use_test_embedder=(not args.use_real_embedder),
            out_json=args.out_json,
            out_md=args.out_md,
        )
    )


if __name__ == "__main__":
    main()
