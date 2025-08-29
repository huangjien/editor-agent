"""Pytest configuration and shared fixtures."""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.config.settings import Settings


@pytest.fixture(scope="session")
def event_loop():
  """Create an instance of the default event loop for the test session."""
  loop = asyncio.get_event_loop_policy().new_event_loop()
  yield loop
  loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
  """Create a temporary directory for tests."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    yield Path(tmp_dir)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
  """Create test settings with temporary directories."""
  return Settings(
    app_name="test-editor-agent",
    app_version="0.1.0",
    debug=True,
    environment="testing",
    log_level="DEBUG",
    workspace_dir=str(temp_dir),
    max_file_size=1024 * 1024,  # 1MB for tests
    database_url="sqlite:///:memory:",
    openai_api_key="test-openai-key",
    anthropic_api_key="test-anthropic-key",
    require_api_key=False,  # Disable API key auth for tests
    api_keys="",
    trusted_hosts="localhost,127.0.0.1,testserver",  # Add testserver for TestClient
  )


@pytest.fixture
def test_settings_with_auth(temp_dir: Path) -> Settings:
  """Create test settings with API key authentication enabled."""
  return Settings(
    app_name="test-editor-agent",
    app_version="0.1.0",
    debug=True,
    environment="testing",
    log_level="DEBUG",
    workspace_dir=str(temp_dir),
    max_file_size=1024 * 1024,  # 1MB for tests
    database_url="sqlite:///:memory:",
    openai_api_key="test-openai-key",
    anthropic_api_key="test-anthropic-key",
    require_api_key=True,  # Enable API key auth for auth tests
    api_keys="test-api-key",
    trusted_hosts="localhost,127.0.0.1,testserver",  # Add testserver for TestClient
  )


@pytest.fixture
def client(test_settings: Settings) -> Generator[TestClient, None, None]:
  """Create a test client for the FastAPI app."""
  from src.main import create_app

  # Create app with test settings
  test_app = create_app(test_settings)

  with TestClient(test_app) as test_client:
    yield test_client


@pytest.fixture
def auth_client(test_settings_with_auth: Settings) -> Generator[TestClient, None, None]:
  """Create a test client with authentication enabled."""
  from src.main import create_app

  # Create app with auth-enabled test settings
  test_app = create_app(test_settings_with_auth)

  with TestClient(test_app) as test_client:
    yield test_client


@pytest_asyncio.fixture
async def async_client(
  test_settings_with_auth: Settings,
) -> AsyncGenerator[AsyncClient, None]:
  """Create an async test client for the FastAPI app."""
  from src.main import create_app
  from httpx import ASGITransport

  # Create app with auth-enabled test settings
  test_app = create_app(test_settings_with_auth)

  async with AsyncClient(
    transport=ASGITransport(app=test_app),
    base_url="http://testserver",
    headers={"host": "testserver"},
  ) as ac:
    yield ac


@pytest.fixture
def mock_openai_client():
  """Mock OpenAI client for testing."""
  mock_client = MagicMock()
  mock_response = MagicMock()
  mock_response.choices = [MagicMock()]
  mock_response.choices[0].message.content = "Test response"
  mock_client.chat.completions.create.return_value = mock_response
  return mock_client


@pytest.fixture
def mock_anthropic_client():
  """Mock Anthropic client for testing."""
  mock_client = MagicMock()
  mock_response = MagicMock()
  mock_response.content = [MagicMock()]
  mock_response.content[0].text = "Test response"
  mock_client.messages.create.return_value = mock_response
  return mock_client


@pytest.fixture
def sample_files(temp_dir: Path) -> dict[str, Path]:
  """Create sample files for testing."""
  files = {}

  # Create a sample text file
  text_file = temp_dir / "sample.txt"
  text_file.write_text("This is a sample text file for testing.")
  files["text"] = text_file

  # Create a sample Python file
  py_file = temp_dir / "sample.py"
  py_file.write_text('def hello():\n    return "Hello, World!"\n')
  files["python"] = py_file

  # Create a sample JSON file
  json_file = temp_dir / "sample.json"
  json_file.write_text('{"key": "value", "number": 42}')
  files["json"] = json_file

  # Create a subdirectory with files
  subdir = temp_dir / "subdir"
  subdir.mkdir()
  sub_file = subdir / "nested.txt"
  sub_file.write_text("Nested file content")
  files["nested"] = sub_file

  return files


@pytest.fixture
def mock_agent_state():
  """Create a mock agent state for testing."""
  from src.agent.state import create_initial_agent_state

  state = create_initial_agent_state(
    session_id="test-session-123", task="Test user input", config={"max_steps": 50}
  )
  # Add some test-specific fields that tests expect
  state["user_input"] = "Test user input"
  state["plan"] = []
  state["actions"] = []
  state["response"] = ""
  state["error"] = None
  state["is_complete"] = False
  state["processing_log"] = []

  return state


@pytest.fixture
def mock_chat_state():
  """Create a mock chat state for testing."""
  from src.agent.state import create_chat_state

  state = create_chat_state(
    session_id="test-chat-123", user_message="Test message", context={}
  )
  # Add test-specific field that tests expect
  state["current_message"] = "Test message"

  return state
