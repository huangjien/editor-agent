"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import (
  Settings,
  get_settings,
)


class TestSettings:
  """Test Settings class."""

  def test_settings_defaults(self):
    """Test Settings with default values."""
    settings = Settings()

    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.debug is True
    assert settings.environment == "development"
    assert settings.app_name == "Editor Agent"
    assert settings.app_version == "1.0.0"

  def test_settings_custom_values(self):
    """Test Settings with custom values."""
    settings = Settings(
      host="127.0.0.1",
      port=9000,
      debug=True,
      environment="production",
      app_name="Custom Agent",
      app_version="2.0.0",
    )

    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.debug is True
    assert settings.environment == "production"
    assert settings.app_name == "Custom Agent"
    assert settings.app_version == "2.0.0"

  def test_settings_validation(self):
    """Test Settings validation."""
    # Test invalid log level
    with pytest.raises(ValidationError):
      Settings(log_level="INVALID")

    # Test invalid environment
    with pytest.raises(ValidationError):
      Settings(environment="invalid")

  def test_settings_properties(self):
    """Test Settings properties."""
    dev_settings = Settings(environment="development")
    prod_settings = Settings(environment="production")

    assert dev_settings.is_development is True
    assert dev_settings.is_production is False

    assert prod_settings.is_development is False
    assert prod_settings.is_production is True

  def test_workspace_path(self):
    """Test workspace path property."""
    settings = Settings(workspace_dir="./test_workspace")
    workspace_path = settings.workspace_path

    assert isinstance(workspace_path, Path)
    assert str(workspace_path).endswith("test_workspace")

  def test_model_config(self):
    """Test model configuration."""
    settings = Settings(
      default_model="gpt-4", model_temperature=0.5, model_max_tokens=2000
    )

    config = settings.get_model_config()

    assert config["model"] == "gpt-4"
    assert config["temperature"] == 0.5
    assert config["max_tokens"] == 2000

  def test_cors_config(self):
    """Test CORS configuration."""
    settings = Settings(
      cors_origins="http://localhost:3000", cors_allow_credentials=True
    )

    config = settings.get_cors_config()

    assert config["allow_origins"] == ["http://localhost:3000"]
    assert config["allow_credentials"] is True

  def test_agent_config(self):
    """Test agent configuration."""
    settings = Settings(max_iterations=20, max_execution_time=600, enable_memory=False)

    config = settings.get_agent_config()

    assert config["max_iterations"] == 20
    assert config["max_execution_time"] == 600
    assert config["enable_memory"] is False

  def test_model_provider_validation(self):
    """Test model provider validation accepts valid providers."""
    # Test valid providers
    for provider in ["openai", "anthropic", "ollama"]:
      settings = Settings(model_provider=provider)
      assert settings.model_provider == provider

    # Test invalid provider
    with pytest.raises(ValidationError, match="Model provider must be one of"):
      Settings(model_provider="invalid_provider")

  def test_ollama_configuration(self):
    """Test Ollama-specific configuration."""
    settings = Settings(
      model_provider="ollama",
      ollama_base_url="http://localhost:11434",
      ollama_model="llama2"
    )

    assert settings.model_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama2"


class TestEnvironmentVariables:
  """Test environment variable loading."""

  def test_settings_from_env_vars(self):
    """Test loading settings from environment variables."""
    env_vars = {
      "APP_NAME": "Test Agent",
      "HOST": "192.168.1.1",
      "PORT": "9999",
      "DEBUG": "true",
      "ENVIRONMENT": "testing",
    }

    with patch.dict(os.environ, env_vars):
      settings = Settings()

      assert settings.app_name == "Test Agent"
      assert settings.host == "192.168.1.1"
      assert settings.port == 9999
      assert settings.debug is True
      assert settings.environment == "testing"

  def test_env_var_type_conversion(self):
    """Test environment variable type conversion."""
    env_vars = {
      "PORT": "8080",
      "DEBUG": "false",
      "MODEL_TEMPERATURE": "0.8",
      "MAX_ITERATIONS": "15",
    }

    with patch.dict(os.environ, env_vars):
      settings = Settings()

      assert isinstance(settings.port, int)
      assert settings.port == 8080
      assert isinstance(settings.debug, bool)
      assert settings.debug is False
      assert isinstance(settings.model_temperature, float)
      assert settings.model_temperature == 0.8
      assert isinstance(settings.max_iterations, int)
      assert settings.max_iterations == 15


class TestGetSettings:
  """Test get_settings function."""

  def test_get_settings_singleton(self):
    """Test that get_settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2

  def test_get_settings_with_env_vars(self):
    """Test get_settings with environment variables."""
    env_vars = {"APP_NAME": "Env Test Agent", "PORT": "7777"}

    with patch.dict(os.environ, env_vars):
      # Clear the cache to ensure fresh settings
      get_settings.cache_clear()
      settings = get_settings()

      assert settings.app_name == "Env Test Agent"
      assert settings.port == 7777


class TestConfigurationIntegration:
  """Test configuration integration scenarios."""

  def test_production_configuration(self):
    """Test production configuration setup."""
    env_vars = {
      "ENVIRONMENT": "production",
      "DEBUG": "false",
      "LOG_LEVEL": "WARNING",
      "RELOAD": "false",
    }

    with patch.dict(os.environ, env_vars):
      settings = Settings()

      assert settings.environment == "production"
      assert settings.debug is False
      assert settings.log_level == "WARNING"
      assert settings.reload is False
      assert settings.is_production is True

  def test_development_configuration(self):
    """Test development configuration setup."""
    env_vars = {
      "ENVIRONMENT": "development",
      "DEBUG": "true",
      "LOG_LEVEL": "DEBUG",
      "RELOAD": "true",
    }

    with patch.dict(os.environ, env_vars):
      settings = Settings()

      assert settings.environment == "development"
      assert settings.debug is True
      assert settings.log_level == "DEBUG"
      assert settings.reload is True
      assert settings.is_development is True

  def test_workspace_path_creation(self):
    """Test workspace path creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
      workspace_dir = Path(temp_dir) / "test_workspace"

      settings = Settings(workspace_dir=str(workspace_dir))
      workspace_path = settings.workspace_path

      assert workspace_path == workspace_dir
      assert workspace_path.parent.exists()  # Parent directory exists
