from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from supportflow.state import BudgetPolicy


class Settings(BaseSettings):
    """SupportFlow 强类型运行配置，统一从环境变量与 .env 读取。"""

    model_config = SettingsConfigDict(
        env_prefix="SUPPORTFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    runtime_mode: Literal["simulation", "live"] = "simulation"

    # LLM 配置
    llm_api_key: SecretStr = Field(default=SecretStr("sk-simulation-placeholder"))
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(default=30.0, gt=0.0)

    # Embedding 配置
    embedding_backend: Literal["sentence_transformers", "openai_compatible"] = (
        "sentence_transformers"
    )
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, ge=64, le=4096)
    embedding_api_base: str | None = None
    embedding_api_key: SecretStr | None = None

    # PostgreSQL 配置
    database_url: str = "postgresql://postgres:postgres@localhost:5432/supportflow"
    db_pool_min_size: int = Field(default=2, ge=1)
    db_pool_max_size: int = Field(default=10, ge=1)
    db_statement_timeout_ms: int = Field(default=5000, ge=100)

    # 默认调查预算与超时策略
    default_max_rounds: int = Field(default=3, ge=1, le=10)
    default_max_tool_attempts: int = Field(default=8, ge=1, le=50)
    default_max_investigation_ms: int = Field(default=45000, ge=1000)
    default_max_agent_steps_per_task: int = Field(default=6, ge=1, le=20)
    default_round_timeout_ms: int = Field(default=20000, ge=1000)
    default_task_timeout_ms: int = Field(default=18000, ge=500)
    default_tool_timeout_ms: int = Field(default=5000, ge=100)

    # 上下文治理阈值
    max_tool_output_chars: int = Field(default=1500, ge=200, le=20000)
    max_main_history_messages: int = Field(default=12, ge=2, le=100)

    # 检索参数
    rrf_k: int = Field(default=60, ge=1, le=200)

    def to_budget_policy(self) -> BudgetPolicy:
        return {
            "max_rounds": self.default_max_rounds,
            "max_tool_attempts": self.default_max_tool_attempts,
            "max_investigation_ms": self.default_max_investigation_ms,
            "max_agent_steps_per_task": self.default_max_agent_steps_per_task,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
