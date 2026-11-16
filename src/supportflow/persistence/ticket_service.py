from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg_pool import AsyncConnectionPool

from supportflow.deps import ClockProtocol, SystemClock, TicketStoreProtocol
from supportflow.schemas import (
    CommitOutcomeUnknownError,
    DefiniteDatabaseError,
    TicketRecord,
    TicketSubmissionOutcome,
)
from supportflow.state import TicketSubmission
from supportflow.tools import FaultInjector


def _extract_title(payload_json: str) -> str:
    try:
        data = json.loads(payload_json)
        return str(data.get("title") or "技术支持工单")
    except (ValueError, TypeError):
        return "技术支持工单"


def _row_to_ticket_record(row: dict[str, Any] | tuple[Any, ...]) -> TicketRecord:
    if isinstance(row, dict):
        created = row["created_at"]
        created_str = (
            created.isoformat() if isinstance(created, datetime) else str(created)
        )
        return TicketRecord(
            id=str(row["id"]),
            account_id=str(row["account_id"]),
            idempotency_key=str(row["idempotency_key"]),
            payload_hash=str(row["payload_hash"]),
            payload_json=str(row["payload_json"]),
            title=str(row["title"]),
            status=str(row["status"]),
            created_at=created_str,
        )
    created = row[7]
    created_str = created.isoformat() if isinstance(created, datetime) else str(created)
    return TicketRecord(
        id=str(row[0]),
        account_id=str(row[1]),
        idempotency_key=str(row[2]),
        payload_hash=str(row[3]),
        payload_json=str(row[4]),
        title=str(row[5]),
        status=str(row[6]),
        created_at=created_str,
    )


class PostgresTicketStore(TicketStoreProtocol):
    """基于 PostgreSQL 事务与唯一约束的幂等工单存储（对应第 16～24 点）。"""

    def __init__(
        self,
        *,
        pool: AsyncConnectionPool[Any],
        fault_injector: FaultInjector | None = None,
        clock: ClockProtocol | None = None,
    ) -> None:
        self._pool = pool
        self._fault_injector = fault_injector
        self._clock = clock or SystemClock()

    async def find_by_idempotency_key(
        self,
        account_id: str,
        idempotency_key: str,
    ) -> TicketRecord | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, account_id, idempotency_key, payload_hash,
                           payload_json, title, status, created_at
                    FROM tickets
                    WHERE account_id = %s AND idempotency_key = %s
                    LIMIT 1
                    """,
                    (account_id, idempotency_key),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_ticket_record(row)

    async def create_idempotent(
        self,
        submission: TicketSubmission,
        account_id: str,
    ) -> TicketSubmissionOutcome:
        idem_key = submission["idempotency_key"]

        # 1) LOOKUP：先按幂等键查询是否已存在
        try:
            existing = await self.find_by_idempotency_key(account_id, idem_key)
        except psycopg.Error as exc:
            return TicketSubmissionOutcome(
                outcome="failed",
                error_message=f"查询已有工单失败: {exc}",
            )

        if existing is not None:
            if existing.payload_hash == submission["payload_hash"]:
                return TicketSubmissionOutcome(outcome="existing", ticket=existing)
            return TicketSubmissionOutcome(
                outcome="conflict",
                error_message="同一幂等键已绑定不同内容的工单，拒绝覆盖创建",
            )

        # 2) TX + COMMIT：开启数据库事务写入工单、幂等键与内容摘要
        new_id = f"T-{uuid.uuid4().hex[:10].upper()}"
        title = _extract_title(submission["payload_json"])
        created_at_iso = self._clock.now_iso()
        committed_record: TicketRecord | None = None

        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    # 若注入的是明确失败，在提交前触发并由事务块自动回滚
                    if self._fault_injector is not None:
                        await self._maybe_inject_pre_commit_error()

                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            INSERT INTO tickets (
                                id, account_id, idempotency_key, payload_hash,
                                payload_json, title, status, created_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, 'open', %s)
                            RETURNING id, account_id, idempotency_key, payload_hash,
                                      payload_json, title, status, created_at
                            """,
                            (
                                new_id,
                                account_id,
                                idem_key,
                                submission["payload_hash"],
                                submission["payload_json"],
                                title,
                                created_at_iso,
                            ),
                        )
                        row = await cur.fetchone()
                        assert row is not None
                        committed_record = _row_to_ticket_record(row)

            # 事务已物理提交后，检查是否模拟“提交成功但响应链路断开（UNKNOWN）”
            if self._fault_injector is not None:
                await self._maybe_inject_post_commit_unknown()

            assert committed_record is not None
            return TicketSubmissionOutcome(outcome="created", ticket=committed_record)

        except pg_errors.UniqueViolation as exc:
            # 唯一约束冲突：async with conn.transaction() 已自动回滚当前事务，执行 ROLLBACK -> LOOKUP
            constraint_name = getattr(getattr(exc, "diag", None), "constraint_name", None)
            if constraint_name and constraint_name != "tickets_idempotency_unique":
                return TicketSubmissionOutcome(
                    outcome="failed",
                    error_message=f"违反非幂等键约束 {constraint_name}: {exc}",
                )
            re_existing = await self.find_by_idempotency_key(account_id, idem_key)
            if re_existing is None:
                return TicketSubmissionOutcome(
                    outcome="failed",
                    error_message="发生唯一约束冲突但回滚重查未命中工单记录",
                )
            if re_existing.payload_hash == submission["payload_hash"]:
                return TicketSubmissionOutcome(outcome="existing", ticket=re_existing)
            return TicketSubmissionOutcome(
                outcome="conflict",
                error_message="并发冲突：同一幂等键已被不同提交内容占用",
            )

        except CommitOutcomeUnknownError as exc:
            return TicketSubmissionOutcome(
                outcome="unknown",
                error_message=str(exc),
            )

        except (DefiniteDatabaseError, psycopg.Error) as exc:
            return TicketSubmissionOutcome(
                outcome="failed",
                error_message=f"数据库写入失败并已回滚: {exc}",
            )

    async def _maybe_inject_pre_commit_error(self) -> None:
        if self._fault_injector is None:
            return
        # 通过临时过滤规则仅检查 db_definite_error（db_commit_unknown 留到提交后触发）
        for rule in self._fault_injector._config.rules:
            if rule.target == "ticket_commit" and rule.fault_type == "db_definite_error":
                await self._fault_injector.maybe_inject_db_fault("ticket_commit")
                return

    async def _maybe_inject_post_commit_unknown(self) -> None:
        if self._fault_injector is None:
            return
        for rule in self._fault_injector._config.rules:
            if rule.target == "ticket_commit" and rule.fault_type == "db_commit_unknown":
                await self._fault_injector.maybe_inject_db_fault("ticket_commit")
                return


class InMemoryTicketStore(TicketStoreProtocol):
    """供无数据库单测/本地仿真使用的内存幂等工单存储。

    内部严格模拟：前置 LOOKUP 窗口并发竞态、唯一约束冲突回滚重查、以及提交后响应丢失（UNKNOWN）。
    """

    def __init__(
        self,
        *,
        fault_injector: FaultInjector | None = None,
        clock: ClockProtocol | None = None,
        simulate_race_yield: bool = True,
    ) -> None:
        self._rows_by_id: dict[str, TicketRecord] = {}
        self._id_by_unique_key: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()
        self._fault_injector = fault_injector
        self._clock = clock or SystemClock()
        self._simulate_race_yield = simulate_race_yield

    def count_tickets(self) -> int:
        return len(self._rows_by_id)

    def list_tickets(self) -> list[TicketRecord]:
        return list(self._rows_by_id.values())

    async def find_by_idempotency_key(
        self,
        account_id: str,
        idempotency_key: str,
    ) -> TicketRecord | None:
        async with self._lock:
            tid = self._id_by_unique_key.get((account_id, idempotency_key))
            return self._rows_by_id.get(tid) if tid is not None else None

    async def create_idempotent(
        self,
        submission: TicketSubmission,
        account_id: str,
    ) -> TicketSubmissionOutcome:
        idem_key = submission["idempotency_key"]

        # 1) LOOKUP
        existing = await self.find_by_idempotency_key(account_id, idem_key)
        if existing is not None:
            if existing.payload_hash == submission["payload_hash"]:
                return TicketSubmissionOutcome(outcome="existing", ticket=existing)
            return TicketSubmissionOutcome(
                outcome="conflict",
                error_message="同一幂等键已绑定不同内容的工单，拒绝覆盖创建",
            )

        # 故意让出一次事件循环，使 10 路并发提交可以真实落入后续唯一约束竞争分支
        if self._simulate_race_yield:
            await asyncio.sleep(0)

        # 2) TX + COMMIT（原子加锁写入唯一索引，若冲突则回滚重查）
        new_id = f"T-{uuid.uuid4().hex[:10].upper()}"
        title = _extract_title(submission["payload_json"])
        record = TicketRecord(
            id=new_id,
            account_id=account_id,
            idempotency_key=idem_key,
            payload_hash=submission["payload_hash"],
            payload_json=submission["payload_json"],
            title=title,
            status="open",
            created_at=self._clock.now_iso()
            if self._clock
            else datetime.now(timezone.utc).isoformat(),
        )

        conflict_detected = False
        try:
            async with self._lock:
                if self._fault_injector is not None:
                    for rule in self._fault_injector._config.rules:
                        if rule.target == "ticket_commit" and rule.fault_type == "db_definite_error":
                            await self._fault_injector.maybe_inject_db_fault("ticket_commit")
                            break

                unique_key = (account_id, idem_key)
                if unique_key in self._id_by_unique_key:
                    conflict_detected = True
                else:
                    self._id_by_unique_key[unique_key] = record.id
                    self._rows_by_id[record.id] = record

            if conflict_detected:
                # ROLLBACK -> LOOKUP
                re_existing = await self.find_by_idempotency_key(account_id, idem_key)
                if re_existing is not None:
                    if re_existing.payload_hash == submission["payload_hash"]:
                        return TicketSubmissionOutcome(outcome="existing", ticket=re_existing)
                    return TicketSubmissionOutcome(
                        outcome="conflict",
                        error_message="并发冲突：同一幂等键已被不同提交内容占用",
                    )
                return TicketSubmissionOutcome(
                    outcome="failed",
                    error_message="唯一约束冲突后重查未找到工单",
                )

            # 事务已提交完成后检查 db_commit_unknown，模拟“落库成功但确认响应丢失”
            if self._fault_injector is not None:
                for rule in self._fault_injector._config.rules:
                    if rule.target == "ticket_commit" and rule.fault_type == "db_commit_unknown":
                        await self._fault_injector.maybe_inject_db_fault("ticket_commit")
                        break

            return TicketSubmissionOutcome(outcome="created", ticket=record)

        except CommitOutcomeUnknownError as exc:
            return TicketSubmissionOutcome(outcome="unknown", error_message=str(exc))
        except DefiniteDatabaseError as exc:
            return TicketSubmissionOutcome(outcome="failed", error_message=str(exc))
