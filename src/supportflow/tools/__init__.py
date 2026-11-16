from __future__ import annotations

import asyncio
from collections import defaultdict

from supportflow.schemas import (
    DefiniteDatabaseError,
    DefiniteToolError,
    FaultInjectorConfig,
    FaultRule,
    TransientToolError,
    CommitOutcomeUnknownError,
)


class FaultInjector:
    """线程/协程安全的定向故障注入器，支撑模拟仿真、跨循环重试与并发工单测试。"""

    def __init__(self, config: FaultInjectorConfig | None = None) -> None:
        self._config = config or FaultInjectorConfig()
        self._attempt_counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    def update_config(self, config: FaultInjectorConfig) -> None:
        self._config = config
        self._attempt_counts.clear()

    def reset_counters(self) -> None:
        self._attempt_counts.clear()

    async def maybe_inject_tool_fault(
        self,
        target: str,
    ) -> FaultRule | None:
        """记录调用尝试并在命中规则时延迟/抛出指定异常，或返回契约损坏指令。"""
        async with self._lock:
            self._attempt_counts[target] += 1
            attempt = self._attempt_counts[target]

        matched: FaultRule | None = None
        for rule in self._config.rules:
            if rule.target != target:
                continue
            if rule.trigger_on_attempt is None or rule.trigger_on_attempt == attempt:
                matched = rule
                break

        if matched is None:
            return None

        if matched.delay_ms > 0:
            await asyncio.sleep(matched.delay_ms / 1000.0)

        if matched.fault_type == "timeout":
            raise TransientToolError(
                f"[{target}] 模拟工具超时 (attempt={attempt}): {matched.detail}",
                kind="timeout",
            )
        if matched.fault_type == "connection_error":
            raise TransientToolError(
                f"[{target}] 模拟网络连接中断 (attempt={attempt}): {matched.detail}",
                kind="connection",
            )
        if matched.fault_type == "definite_tool_error":
            raise DefiniteToolError(
                f"[{target}] 模拟确定性工具失败 (attempt={attempt}): {matched.detail}"
            )
        # invalid_output_schema 由调用方根据返回的 rule 构造缺字段的畸形返回体
        return matched

    async def maybe_inject_db_fault(self, target: str = "ticket_commit") -> None:
        """在工单写入提交阶段注入 CommitOutcomeUnknownError 或 DefiniteDatabaseError。"""
        async with self._lock:
            self._attempt_counts[target] += 1
            attempt = self._attempt_counts[target]

        matched: FaultRule | None = None
        for rule in self._config.rules:
            if rule.target != target:
                continue
            if rule.trigger_on_attempt is None or rule.trigger_on_attempt == attempt:
                matched = rule
                break

        if matched is None:
            return

        if matched.delay_ms > 0:
            await asyncio.sleep(matched.delay_ms / 1000.0)

        if matched.fault_type == "db_commit_unknown":
            raise CommitOutcomeUnknownError(
                f"[{target}] 模拟数据库提交后响应丢失 (attempt={attempt}): {matched.detail}"
            )
        if matched.fault_type == "db_definite_error":
            raise DefiniteDatabaseError(
                f"[{target}] 模拟数据库明确失败 (attempt={attempt}): {matched.detail}"
            )


__all__ = ["FaultInjector"]
