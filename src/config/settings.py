"""Configuration settings for the editor agent application."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  """Application settings loaded from environment variables."""

  # Application settings
  app_name: str = Field(default="Editor Agent")
  app_version: str = Field(default="1.0.0")
  debug: bool = Field(default=False)
  environment: str = Field(default="development")

  # Server settings
  host: str = Field(default="0.0.0.0")
  port: int = Field(default=8000)
  reload: bool = Field(default=True)
  workers: int = Field(default=1)

  # API settings
  api_prefix: str = Field(default="/api/v1")
  cors_origins: str = Field(
    default="http://localhost:3000,http://localhost:8080"
  )
  cors_allow_credentials: bool = Field(default=True)
  cors_allow_methods: str = Field(
    default="GET,POST,PUT,DELETE,OPTIONS"
  )
  cors_allow_headers: str = Field(default="*")

  # Security settings
  secret_key: str = Field(default="your-secret-key-change-in-production")
  access_token_expire_minutes: int = Field(default=30)
  trusted_hosts: str = Field(default="localhost,127.0.0.1")

  # LangGraph/LangChain settings
  openai_api_key: Optional[str] = Field(default=None)
  anthropic_api_key: Optional[str] = Field(default=None)
  ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
  ollama_model: str = Field(default="llama2", description="Ollama model name")
  langchain_api_key: Optional[str] = Field(default=None)
  langchain_project: Optional[str] = Field(default=None)
  langchain_tracing_v2: bool = Field(default=False)

  # Model settings
  default_model: str = Field(default="gpt-3.5-turbo")
  model_provider: str = Field(default="openai")
  model_temperature: float = Field(default=0.7)
  model_max_tokens: int = Field(default=1000)
  model_timeout: int = Field(default=30)

  # Agent settings
  max_iterations: int = Field(default=10)
  max_execution_time: int = Field(default=300)  # 5 minutes
  enable_memory: bool = Field(default=True)
  memory_max_tokens: int = Field(default=4000)

  # File system settings
  workspace_dir: str = Field(default="./workspace")
  max_file_size: int = Field(default=10 * 1024 * 1024)  # 10MB
  allowed_file_extensions: str = Field(
    default=".py,.js,.ts,.html,.css,.json,.yaml,.yml,.md,.txt"
  )

  # Logging settings
  log_level: str = Field(default="INFO")
  log_format: str = Field(
    default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  )
  log_file: Optional[str] = Field(default=None)
  log_rotation: str = Field(default="1 day")
  log_retention: str = Field(default="30 days")

  # Database settings - Supabase
  supabase_url: Optional[str] = Field(default=None, description="Supabase project URL")
  supabase_key: Optional[str] = Field(default=None, description="Supabase anon/service role key")
  supabase_service_key: Optional[str] = Field(default=None, description="Supabase service role key for admin operations")
  database_url: Optional[str] = Field(default=None, description="Direct database URL (alternative to Supabase)")
  redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")

  # Monitoring settings
  enable_metrics: bool = Field(default=False)
  metrics_port: int = Field(default=9090)
  health_check_interval: int = Field(default=30)

  # Rate limiting settings
  rate_limit_enabled: bool = Field(default=True)
  rate_limit_requests: int = Field(default=100)
  rate_limit_window: int = Field(default=60)  # seconds
  rate_limit_requests_per_minute: int = Field(default=60)

  # API Key authentication settings
  require_api_key: bool = Field(default=False)
  api_keys: str = Field(default="")

  # Request Size Limit
  max_request_size: int = Field(default=10485760)  # 10MB

  @field_validator("cors_origins", mode="after")
  @classmethod
  def parse_cors_origins(cls, v):
    if isinstance(v, str):
      return [origin.strip() for origin in v.split(",") if origin.strip()]
    return v

  @field_validator("cors_allow_methods", mode="after")
  @classmethod
  def parse_cors_methods(cls, v):
    if isinstance(v, str):
      return [method.strip() for method in v.split(",") if method.strip()]
    return v

  @field_validator("cors_allow_headers", mode="after")
  @classmethod
  def parse_cors_headers(cls, v):
    if isinstance(v, str):
      return [header.strip() for header in v.split(",") if header.strip()]
    return v

  @field_validator("trusted_hosts", mode="after")
  @classmethod
  def parse_trusted_hosts(cls, v):
    if isinstance(v, str):
      return [host.strip() for host in v.split(",") if host.strip()]
    return v

  @field_validator("allowed_file_extensions", mode="after")
  @classmethod
  def parse_file_extensions(cls, v):
    if isinstance(v, str):
      return [ext.strip() for ext in v.split(",") if ext.strip()]
    return v

  @field_validator("api_keys", mode="after")
  @classmethod
  def parse_api_keys(cls, v):
    if isinstance(v, str) and v.strip():
      return [key.strip() for key in v.split(",") if key.strip()]
    return []

  @field_validator("log_level")
  @classmethod
  def validate_log_level(cls, v):
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if v.upper() not in valid_levels:
      raise ValueError(f"Log level must be one of: {valid_levels}")
    return v.upper()

  @field_validator("environment")
  @classmethod
  def validate_environment(cls, v):
    valid_envs = ["development", "staging", "production", "testing"]
    if v.lower() not in valid_envs:
      raise ValueError(f"Environment must be one of: {valid_envs}")
    return v.lower()

  @field_validator("model_temperature")
  @classmethod
  def validate_temperature(cls, v):
    if not 0.0 <= v <= 2.0:
      raise ValueError("Model temperature must be between 0.0 and 2.0")
    return v

  @field_validator("model_provider")
  @classmethod
  def validate_model_provider(cls, v):
    valid_providers = ["openai", "anthropic", "ollama"]
    if v.lower() not in valid_providers:
      raise ValueError(f"Model provider must be one of: {valid_providers}")
    return v.lower()

  @field_validator("workspace_dir")
  @classmethod
  def validate_workspace_dir(cls, v):
    # Ensure workspace directory exists
    Path(v).mkdir(parents=True, exist_ok=True)
    return v

  @property
  def is_development(self) -> bool:
    """Check if running in development environment."""
    return self.environment == "development"

  @property
  def is_production(self) -> bool:
    """Check if running in production environment."""
    return self.environment == "production"

  @property
  def workspace_path(self) -> Path:
    """Get workspace directory as Path object."""
    return Path(self.workspace_dir)

  def get_model_config(self) -> Dict[str, Any]:
    """Get model configuration dictionary."""
    return {
      "model": self.default_model,
      "temperature": self.model_temperature,
      "max_tokens": self.model_max_tokens,
      "timeout": self.model_timeout,
    }

  def get_cors_config(self) -> Dict[str, Any]:
    """Get CORS configuration dictionary."""
    return {
      "allow_origins": self.cors_origins,
      "allow_credentials": self.cors_allow_credentials,
      "allow_methods": self.cors_allow_methods,
      "allow_headers": self.cors_allow_headers,
    }

  def get_agent_config(self) -> Dict[str, Any]:
    """Get agent configuration dictionary."""
    return {
      "max_iterations": self.max_iterations,
      "max_execution_time": self.max_execution_time,
      "enable_memory": self.enable_memory,
      "memory_max_tokens": self.memory_max_tokens,
      "workspace_dir": self.workspace_dir,
      "max_file_size": self.max_file_size,
      "allowed_extensions": self.allowed_file_extensions,
    }

  model_config = {
    "env_file": ".env",
    "env_file_encoding": "utf-8",
    "case_sensitive": False,
  }


@lru_cache()
def get_settings() -> Settings:
  """Get cached settings instance."""
  return Settings()


def load_settings_from_file(file_path: str) -> Settings:
  """Load settings from a specific file."""
  return Settings(_env_file=file_path)


def get_env_info() -> Dict[str, Any]:
  """Get environment information for debugging."""
  settings = get_settings()
  return {
    "app_name": settings.app_name,
    "app_version": settings.app_version,
    "environment": settings.environment,
    "debug": settings.debug,
    "python_version": os.sys.version,
    "working_directory": os.getcwd(),
    "workspace_directory": str(settings.workspace_path),
    "log_level": settings.log_level,
  }
