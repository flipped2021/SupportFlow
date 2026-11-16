from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from supportflow.deps import ClockProtocol, OperationsGatewayProtocol, SystemClock
from supportflow.schemas import DefiniteToolError, TransientToolError
from supportflow.tools import FaultInjector


def default_saas_accounts() -> dict[str, dict[str, Any]]:
    """预置模拟文件存储 SaaS 账户快照（用于本地联调、评测与故障压测）。"""
    return {
        "acc-001": {
            "tier": "pro",
            "status": "active",
            "permissions": ["storage:read", "storage:write"],
            "token_scopes": ["storage:read"],
            "can_upload": False,
            "storage_used_bytes": 4_200_000_000,
            "storage_limit_bytes": 50_000_000_000,
            "daily_api_calls": 1200,
            "daily_api_limit": 100_000,
            "logs": [
                {
                    "request_id": "req-123",
                    "timestamp": "2026-11-16T09:15:00Z",
                    "endpoint": "POST /api/v2/files/upload",
                    "http_status": 403,
                    "error_code": "E_UPLOAD_403",
                    "reason": "token missing required scope storage:write under STRICT_SCOPE_CHECK_V23",
                    "product_version": "v2.3",
                },
                {
                    "request_id": "req-120",
                    "timestamp": "2026-11-16T08:40:00Z",
                    "endpoint": "GET /api/v2/files/list",
                    "http_status": 200,
                    "error_code": None,
                    "reason": "ok",
                    "product_version": "v2.3",
                },
            ],
        },
        "acc-002": {
            "tier": "starter",
            "status": "active",
            "permissions": ["storage:read", "storage:write"],
            "token_scopes": ["storage:read", "storage:write"],
            "can_upload": True,
            "storage_used_bytes": 10_737_418_240,
            "storage_limit_bytes": 10_737_418_240,
            "daily_api_calls": 350,
            "daily_api_limit": 10_000,
            "logs": [
                {
                    "request_id": "req-202",
                    "timestamp": "2026-11-16T10:02:00Z",
                    "endpoint": "POST /api/v2/files/upload",
                    "http_status": 403,
                    "error_code": "E_QUOTA_STORAGE_FULL",
                    "reason": "account storage_used_bytes reached storage_limit_bytes",
                    "product_version": "v2.3",
                }
            ],
        },
        "acc-003": {
            "tier": "enterprise",
            "status": "active",
            "permissions": ["storage:read", "storage:write"],
            "token_scopes": ["storage:read", "storage:write"],
            "can_upload": True,
            "storage_used_bytes": 1_000_000_000,
            "storage_limit_bytes": 500_000_000_000,
            "daily_api_calls": 4800,
            "daily_api_limit": 500_000,
            "logs": [
                {
                    "request_id": "req-303",
                    "timestamp": "2026-11-16T11:20:00Z",
                    "endpoint": "POST /api/v2/files/multipart/init",
                    "http_status": 503,
                    "error_code": "E_GATEWAY_REGION_DEGRADED",
                    "reason": "cn-south-1 upload gateway degraded (INC-2026-1104)",
                    "product_version": "v2.3",
                }
            ],
        },
        "acc-004": {
            "tier": "pro",
            "status": "suspended",
            "permissions": ["storage:read"],
            "token_scopes": ["storage:read"],
            "can_upload": False,
            "storage_used_bytes": 2_000_000_000,
            "storage_limit_bytes": 50_000_000_000,
            "daily_api_calls": 10,
            "daily_api_limit": 100_000,
            "logs": [
                {
                    "request_id": "req-404",
                    "timestamp": "2026-11-16T07:10:00Z",
                    "endpoint": "POST /api/v2/files/upload",
                    "http_status": 403,
                    "error_code": "E_ACCOUNT_RESTRICTED",
                    "reason": "account status is suspended due to billing overdue",
                    "product_version": "v2.3",
                }
            ],
        },
    }


def default_service_statuses() -> dict[tuple[str, str], dict[str, Any]]:
    return {
        ("upload", "cn-south-1"): {
            "health": "healthy",
            "incident_id": None,
            "summary": "cn-south-1 upload service operating normally",
        },
        ("auth", "cn-south-1"): {
            "health": "healthy",
            "incident_id": None,
            "summary": "cn-south-1 auth token verification operating normally",
        },
        ("storage", "cn-south-1"): {
            "health": "healthy",
            "incident_id": None,
            "summary": "cn-south-1 object storage operating normally",
        },
        ("gateway", "cn-south-1"): {
            "health": "healthy",
            "incident_id": None,
            "summary": "cn-south-1 edge gateway operating normally",
        },
        ("upload", "cn-east-1"): {
            "health": "degraded",
            "incident_id": "INC-2026-1104",
            "summary": "cn-east-1 upload gateway experiencing elevated 503 latency",
        },
    }


class SimulationOperationsGateway(OperationsGatewayProtocol):
    """4 类只读模拟业务工具网关，支持确定性仿真与故障注入。"""

    def __init__(
        self,
        *,
        accounts: dict[str, dict[str, Any]] | None = None,
        service_statuses: dict[tuple[str, str], dict[str, Any]] | None = None,
        fault_injector: FaultInjector | None = None,
        clock: ClockProtocol | None = None,
    ) -> None:
        self._accounts = accounts if accounts is not None else default_saas_accounts()
        self._service_statuses = (
            service_statuses if service_statuses is not None else default_service_statuses()
        )
        self._fault_injector = fault_injector
        self._clock = clock or SystemClock()

    async def invoke_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(max(0.01, timeout_s)):
                if self._fault_injector is not None:
                    rule = await self._fault_injector.maybe_inject_tool_fault(tool_name)
                    if rule is not None and rule.fault_type == "invalid_output_schema":
                        return {"malformed": True, "tool": tool_name}
                return self._execute_readonly_tool(tool_name, args)
        except TimeoutError as exc:
            raise TransientToolError(
                f"工具 {tool_name} 调用超过单次限制 {timeout_s:.2f}s",
                kind="timeout",
            ) from exc

    def _execute_readonly_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        checked_at = self._clock.now_iso()

        if tool_name == "get_account_permissions":
            account_id = args["account_id"]
            acc = self._accounts.get(account_id)
            if acc is None:
                raise DefiniteToolError(f"账户 {account_id} 不存在")
            return {
                "account_id": account_id,
                "tier": acc["tier"],
                "status": acc["status"],
                "permissions": list(acc["permissions"]),
                "token_scopes": list(acc["token_scopes"]),
                "can_upload": bool(acc["can_upload"]),
                "checked_at": checked_at,
            }

        if tool_name == "get_account_quota":
            account_id = args["account_id"]
            acc = self._accounts.get(account_id)
            if acc is None:
                raise DefiniteToolError(f"账户 {account_id} 不存在")
            used = int(acc["storage_used_bytes"])
            limit_bytes = int(acc["storage_limit_bytes"])
            api_calls = int(acc["daily_api_calls"])
            api_limit = int(acc["daily_api_limit"])
            return {
                "account_id": account_id,
                "storage_used_bytes": used,
                "storage_limit_bytes": limit_bytes,
                "daily_api_calls": api_calls,
                "daily_api_limit": api_limit,
                "quota_exceeded": (used >= limit_bytes) or (api_calls >= api_limit),
                "checked_at": checked_at,
            }

        if tool_name == "get_service_status":
            service_name = args["service_name"]
            region = args.get("region", "cn-south-1")
            if service_name == "all":
                degraded = [
                    (svc, st)
                    for (svc, reg), st in self._service_statuses.items()
                    if reg == region and st["health"] != "healthy"
                ]
                if degraded:
                    svc, st = degraded[0]
                    return {
                        "service_name": f"all(degraded:{svc})",
                        "region": region,
                        "health": st["health"],
                        "incident_id": st["incident_id"],
                        "summary": st["summary"],
                        "checked_at": checked_at,
                    }
                return {
                    "service_name": "all",
                    "region": region,
                    "health": "healthy",
                    "incident_id": None,
                    "summary": f"all services in {region} operating normally",
                    "checked_at": checked_at,
                }

            st = self._service_statuses.get(
                (service_name, region),
                {
                    "health": "healthy",
                    "incident_id": None,
                    "summary": f"{service_name} in {region} operating normally",
                },
            )
            return {
                "service_name": service_name,
                "region": region,
                "health": st["health"],
                "incident_id": st["incident_id"],
                "summary": st["summary"],
                "checked_at": checked_at,
            }

        if tool_name == "get_request_logs":
            account_id = args["account_id"]
            acc = self._accounts.get(account_id)
            if acc is None:
                raise DefiniteToolError(f"账户 {account_id} 不存在")

            request_id = args.get("request_id")
            start_dt = (
                datetime.fromisoformat(args["start_time"])
                if args.get("start_time")
                else None
            )
            end_dt = (
                datetime.fromisoformat(args["end_time"])
                if args.get("end_time")
                else None
            )
            limit = int(args.get("limit", 10))

            matched: list[dict[str, Any]] = []
            for entry in acc["logs"]:
                if request_id is not None and entry["request_id"] != request_id:
                    continue
                entry_dt = datetime.fromisoformat(entry["timestamp"])
                if start_dt is not None and entry_dt < start_dt:
                    continue
                if end_dt is not None and entry_dt > end_dt:
                    continue
                matched.append(dict(entry))

            matched.sort(key=lambda item: item["timestamp"], reverse=True)
            return {
                "account_id": account_id,
                "entries": matched[:limit],
                "checked_at": checked_at,
            }

        raise DefiniteToolError(f"未知业务只读工具: {tool_name}")
