"""
Configuration module for the AI Gateway.
Loads settings from environment variables and defines pricing, model tiers, and policies.
"""

from enum import Enum
from typing import Dict, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict


class RoutePolicy(str, Enum):
    AUTO_ROUTER = "auto_router"
    ALWAYS_CHEAP = "always_cheap"
    ALWAYS_STRONG = "always_strong"


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API Keys & URLs
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Model identifiers
    CHEAP_MODEL: str = "llama-3.1-8b-instant"
    STRONG_MODEL: str = "llama-3.3-70b-versatile"
    FALLBACK_MODEL: str = "llama-3.1-8b-instant"

    # Resilience
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: float = 15.0
    RETRY_BACKOFF_FACTOR: float = 1.5

    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 300

    # Storage
    DATABASE_PATH: str = "gateway.db"

    # Router
    COMPLEXITY_THRESHOLD: float = 0.30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Pricing table: (Cost per 1M prompt tokens, Cost per 1M completion tokens) in USD
# Derived from official pricing (e.g., Groq / OpenAI)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Groq / Open source LLaMA models
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.2-1b-preview": (0.04, 0.04),
    "llama-3.2-3b-preview": (0.06, 0.06),
    "mixtral-8x7b-32768": (0.24, 0.24),

    # OpenAI models
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-3.5-turbo": (0.50, 1.50),

    # Built-in Mock / Test models
    "mock-cheap": (0.05, 0.08),
    "mock-strong": (0.59, 0.79),
    "default": (0.10, 0.20),
}
