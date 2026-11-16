from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from supportflow.config import Settings
from supportflow.env_factory import SimulationChatModel, create_simulation_deps
from supportflow.graph import build_support_graph
from supportflow.persistence.ticket_service import InMemoryTicketStore
from supportflow.schemas import FaultInjectorConfig


@pytest.fixture
def sim_settings() -> Settings:
    return Settings(
        runtime_mode="simulation",
        default_max_rounds=3,
        default_max_tool_attempts=8,
        default_max_investigation_ms=30000,
        default_max_agent_steps_per_task=6,
        default_round_timeout_ms=15000,
        default_task_timeout_ms=12000,
        default_tool_timeout_ms=3000,
    )


@pytest.fixture
def memory_graph():
    checkpointer = MemorySaver()
    return build_support_graph(checkpointer=checkpointer)


@pytest.fixture
def make_deps(sim_settings: Settings):
    def _factory(
        *,
        fault_config: FaultInjectorConfig | None = None,
        verified_account_id: str | None = None,
        llm: SimulationChatModel | None = None,
        ticket_store: InMemoryTicketStore | None = None,
    ):
        return create_simulation_deps(
            settings=sim_settings,
            fault_config=fault_config,
            verified_account_id=verified_account_id,
            llm=llm or SimulationChatModel(),
            use_test_embedder=True,
            ticket_store=ticket_store,
        )

    return _factory
