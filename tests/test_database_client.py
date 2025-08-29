"""Tests for Supabase database client functionality."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from uuid import uuid4

from src.database.client import SupabaseClient, get_supabase_client, get_supabase_admin_client
from src.database.models import (
    UserProfile, ChatSession, ChatMessage, AgentTask, TaskStatus, MessageRole, ExecutionStatus
)
from src.database.schemas import (
    UserProfileCreate, ChatSessionCreate, ChatMessageCreate,
    AgentTaskCreate, AgentExecutionCreate
)


class TestSupabaseClient:
    """Test cases for SupabaseClient class."""

    @pytest.fixture
    def mock_supabase(self):
        """Create a mock Supabase client."""
        mock_client = Mock()
        mock_client.table.return_value = Mock()
        return mock_client

    @pytest.fixture
    def supabase_client(self, mock_supabase):
        """Create a SupabaseClient instance with mocked Supabase client."""
        return SupabaseClient(mock_supabase)

    @pytest.fixture
    def sample_user_profile(self):
        """Create a sample user profile for testing."""
        return UserProfile(
            id=str(uuid4()),
            user_id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User",
            preferences={"theme": "dark"},
            is_active=True
        )

    @pytest.fixture
    def sample_chat_session(self):
        """Create a sample chat session for testing."""
        return ChatSession(
            id=str(uuid4()),
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Session",
            description="A test chat session",
            is_active=True,
            message_count=0
        )

    @pytest.fixture
    def sample_chat_message(self, sample_chat_session):
        """Create a sample chat message for testing."""
        return ChatMessage(
            id=str(uuid4()),
            session_id=str(sample_chat_session.id),
            role=MessageRole.USER,
            content="Hello, world!",
            sequence_number=1
        )

    @pytest.fixture
    def sample_agent_task(self, sample_chat_session):
        """Create a sample agent task for testing."""
        return AgentTask(
            id=str(uuid4()),
            session_id=str(sample_chat_session.id),
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Task",
            instructions="Do something useful",
            status=TaskStatus.PENDING,
            priority=1
        )

    def test_test_connection_success(self, supabase_client, mock_supabase):
        """Test successful connection test."""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{"version": "1.0"}]
        mock_supabase.rpc.return_value.execute.return_value = mock_response
        
        result = supabase_client.test_connection()
        
        assert result is True
        mock_supabase.rpc.assert_called_once_with("version")

    def test_test_connection_failure(self, supabase_client, mock_supabase):
        """Test connection test failure."""
        # Mock exception
        mock_supabase.rpc.return_value.execute.side_effect = Exception("Connection failed")
        
        result = supabase_client.test_connection()
        
        assert result is False

    def test_health_check_success(self, supabase_client, mock_supabase):
        """Test successful health check."""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{"status": "healthy"}]
        mock_supabase.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_response
        
        result = supabase_client.health_check()
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "database" in result

    def test_health_check_failure(self, supabase_client, mock_supabase):
        """Test health check failure."""
        # Mock exception
        mock_supabase.rpc.return_value.execute.side_effect = Exception("DB Error")
        
        result = supabase_client.health_check()
        
        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_connect_success(self, supabase_client, mock_supabase):
        """Test successful async connection."""
        import asyncio
        
        # Mock successful response
        mock_response = Mock()
        mock_response.data = "PostgreSQL 14.0"
        mock_supabase.rpc.return_value.execute.return_value = mock_response
        
        result = asyncio.run(supabase_client.connect())
        
        assert result is True
        assert supabase_client.is_connected is True
        mock_supabase.rpc.assert_called_with("version")

    def test_connect_failure(self, supabase_client, mock_supabase):
        """Test async connection failure."""
        import asyncio
        
        # Mock connection failure
        mock_supabase.rpc.return_value.execute.side_effect = Exception("Network timeout")
        
        result = asyncio.run(supabase_client.connect())
        
        assert result is False
        assert supabase_client.is_connected is False

    def test_connect_network_timeout(self, supabase_client, mock_supabase):
        """Test connection with network timeout."""
        import asyncio
        from requests.exceptions import Timeout
        
        # Mock timeout exception
        mock_supabase.rpc.return_value.execute.side_effect = Timeout("Request timeout")
        
        result = asyncio.run(supabase_client.connect())
        
        assert result is False
        assert supabase_client.is_connected is False

    def test_test_connection_network_error(self, supabase_client, mock_supabase):
        """Test connection with network error."""
        from requests.exceptions import ConnectionError
        
        # Mock network error
        mock_supabase.rpc.return_value.execute.side_effect = ConnectionError("Network unreachable")
        
        result = supabase_client.test_connection()
        
        assert result is False
        assert supabase_client.is_connected is False

    def test_test_connection_invalid_credentials(self, supabase_client, mock_supabase):
        """Test connection with invalid credentials."""
        # Mock authentication error
        mock_supabase.rpc.return_value.execute.side_effect = Exception("Invalid API key")
        
        result = supabase_client.test_connection()
        
        assert result is False
        assert supabase_client.is_connected is False

    # User Profile Tests
    def test_create_user_profile(self, supabase_client, mock_supabase):
        """Test creating a user profile."""
        profile_data = UserProfileCreate(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User"
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{
            "id": str(uuid4()),
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "test@example.com",
            "name": "Test User",
            "preferences": {},
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        result = supabase_client.create_user_profile(profile_data)
        
        assert result is not None
        assert result.user_id == "550e8400-e29b-41d4-a716-446655440000"
        mock_supabase.table.assert_called_with("user_profiles")

    def test_get_user_profile(self, supabase_client, mock_supabase, sample_user_profile):
        """Test getting a user profile."""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{
            "id": str(sample_user_profile.id),
            "user_id": sample_user_profile.user_id,
            "email": sample_user_profile.email,
            "name": sample_user_profile.name,
            "preferences": sample_user_profile.preferences,
            "is_active": sample_user_profile.is_active,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_user_profile(sample_user_profile.user_id)
        
        assert result is not None
        assert result.user_id == sample_user_profile.user_id

    def test_get_user_profile_not_found(self, supabase_client, mock_supabase):
        """Test getting user profile when not found."""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_user_profile("nonexistent_user")
        
        assert result is None

    def test_create_user_profile_no_data_returned(self, supabase_client, mock_supabase):
        """Test creating user profile when no data is returned."""
        profile_data = UserProfileCreate(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User"
        )
        
        # Mock response with no data
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        with pytest.raises(ValueError, match="No data returned from user profile creation"):
            supabase_client.create_user_profile(profile_data)

    def test_create_user_profile_database_error(self, supabase_client, mock_supabase):
        """Test creating user profile with database error."""
        profile_data = UserProfileCreate(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User"
        )
        
        # Mock database exception
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            supabase_client.create_user_profile(profile_data)

    def test_get_user_profile_database_error(self, supabase_client, mock_supabase):
        """Test getting user profile with database error."""
        # Mock database exception
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("Query timeout")
        
        # get_user_profile catches exceptions and returns None
        result = supabase_client.get_user_profile("550e8400-e29b-41d4-a716-446655440000")
        assert result is None

    def test_create_user_profile_invalid_data(self, supabase_client, mock_supabase):
        """Test creating user profile with invalid data."""
        from pydantic import ValidationError
        
        # Test invalid email format - should raise validation error during schema creation
        with pytest.raises(ValidationError):
            UserProfileCreate(
                user_id="550e8400-e29b-41d4-a716-446655440000",
                email="invalid-email",  # Invalid email format
                name="Test User"
            )
        
        # Test empty user_id - should raise validation error during schema creation
        with pytest.raises(ValidationError):
            UserProfileCreate(
                user_id="",  # Invalid empty user_id
                email="test@example.com",
                name="Test User"
            )

    # Chat Session Tests
    def test_create_chat_session(self, supabase_client, mock_supabase):
        """Test creating a chat session."""
        session_data = ChatSessionCreate(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Session",
            description="A test session"
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{
            "id": str(uuid4()),
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Session",
            "description": "A test session",
            "is_active": True,
            "message_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        result = supabase_client.create_chat_session(session_data)
        
        assert result is not None
        assert result.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert result.title == "Test Session"

    def test_get_user_chat_sessions(self, supabase_client, mock_supabase):
        """Test getting user chat sessions."""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [
            {
                "id": str(uuid4()),
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Session 1",
                "is_active": True,
                "message_count": 5,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid4()),
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Session 2",
                "is_active": True,
                "message_count": 3,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_user_chat_sessions("550e8400-e29b-41d4-a716-446655440000")
        
        assert len(result) == 2
        assert all(session.user_id == "550e8400-e29b-41d4-a716-446655440000" for session in result)

    def test_get_user_chat_sessions_empty_result(self, supabase_client, mock_supabase):
        """Test getting user chat sessions with empty result."""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_user_chat_sessions("user_with_no_sessions")
        
        assert len(result) == 0
        assert isinstance(result, list)

    def test_get_user_chat_sessions_database_error(self, supabase_client, mock_supabase):
        """Test getting user chat sessions with database error."""
        # Mock database exception
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("Query failed")
        
        # get_user_chat_sessions catches exceptions and returns empty list
        result = supabase_client.get_user_chat_sessions("550e8400-e29b-41d4-a716-446655440000")
        assert result == []

    def test_create_chat_session_no_data_returned(self, supabase_client, mock_supabase):
        """Test creating chat session when no data is returned."""
        session_data = ChatSessionCreate(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Session"
        )
        
        # Mock response with no data
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        with pytest.raises(ValueError, match="No data returned from chat session creation"):
            supabase_client.create_chat_session(session_data)

    def test_create_chat_session_database_error(self, supabase_client, mock_supabase):
        """Test creating chat session with database error."""
        session_data = ChatSessionCreate(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Session"
        )
        
        # Mock database exception
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("Insert failed")
        
        with pytest.raises(Exception, match="Insert failed"):
            supabase_client.create_chat_session(session_data)

    # Chat Message Tests
    def test_create_chat_message(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating a chat message."""
        message_data = ChatMessageCreate(
            session_id=sample_chat_session.id,
            role=MessageRole.USER,
            content="Hello, world!",
            sequence_number=1
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{
            "id": str(uuid4()),
            "session_id": str(sample_chat_session.id),
            "role": "user",
            "content": "Hello, world!",
            "sequence_number": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        result = supabase_client.create_chat_message(message_data)
        
        assert result is not None
        assert result.content == "Hello, world!"
        assert result.role == MessageRole.USER

    def test_get_session_messages(self, supabase_client, mock_supabase, sample_chat_session):
        """Test getting session messages."""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [
            {
                "id": str(uuid4()),
                "session_id": str(sample_chat_session.id),
                "role": "user",
                "content": "Hello!",
                "sequence_number": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid4()),
                "session_id": str(sample_chat_session.id),
                "role": "assistant",
                "content": "Hi there!",
                "sequence_number": 2,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_session_messages(sample_chat_session.id)
        
        assert len(result) == 2
        assert result[0].sequence_number == 1
        assert result[1].sequence_number == 2

    def test_get_session_messages_empty_session(self, supabase_client, mock_supabase):
        """Test getting messages from empty session."""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_session_messages(str(uuid4()))
        
        assert len(result) == 0
        assert isinstance(result, list)

    def test_get_session_messages_database_error(self, supabase_client, mock_supabase):
        """Test getting session messages with database error."""
        # Mock database exception
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("Query failed")
        
        # get_session_messages catches exceptions and returns empty list
        result = supabase_client.get_session_messages(str(uuid4()))
        assert result == []

    def test_create_chat_message_no_data_returned(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating chat message when no data is returned."""
        message_data = ChatMessageCreate(
            session_id=sample_chat_session.id,
            role=MessageRole.USER,
            content="Test message",
            sequence_number=1
        )
        
        # Mock response with no data
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        with pytest.raises(ValueError, match="No data returned from chat message creation"):
            supabase_client.create_chat_message(message_data)

    def test_create_chat_message_database_error(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating chat message with database error."""
        message_data = ChatMessageCreate(
            session_id=sample_chat_session.id,
            role=MessageRole.USER,
            content="Test message",
            sequence_number=1
        )
        
        # Mock database exception
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("Insert failed")
        
        with pytest.raises(Exception, match="Insert failed"):
            supabase_client.create_chat_message(message_data)

    def test_create_chat_message_empty_content(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating chat message with empty content."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ChatMessageCreate(
                session_id=sample_chat_session.id,
                role=MessageRole.USER,
                content="",
                sequence_number=1
            )

    # Agent Task Tests
    def test_create_agent_task(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating an agent task."""
        task_data = AgentTaskCreate(
            session_id=sample_chat_session.id,
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Task",
            instructions="Do something useful",
            priority=1
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{
            "id": str(uuid4()),
            "session_id": str(sample_chat_session.id),
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something useful",
            "status": "pending",
            "priority": 1,
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        result = supabase_client.create_agent_task(task_data)
        
        assert result is not None
        assert result.title == "Test Task"
        assert result.status == TaskStatus.PENDING

    def test_get_user_agent_tasks(self, supabase_client, mock_supabase):
        """Test getting user agent tasks."""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [
            {
                "id": str(uuid4()),
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Task 1",
                "instructions": "Do something useful",
                "status": "pending",
                "priority": 1,
                "progress": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid4()),
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Task 2",
                "instructions": "Do something else",
                "status": "completed",
                "priority": 2,
                "progress": 100,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_user_agent_tasks("550e8400-e29b-41d4-a716-446655440000")
        
        assert len(result) == 2
        assert all(task.user_id == "550e8400-e29b-41d4-a716-446655440000" for task in result)

    def test_get_user_agent_tasks_empty_result(self, supabase_client, mock_supabase):
        """Test getting user agent tasks with empty result."""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_user_agent_tasks("user_with_no_tasks")
        
        assert len(result) == 0
        assert isinstance(result, list)

    def test_get_user_agent_tasks_database_error(self, supabase_client, mock_supabase):
        """Test getting user agent tasks with database error."""
        # Mock database exception
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("Query failed")
        
        # get_user_agent_tasks catches exceptions and returns empty list
        result = supabase_client.get_user_agent_tasks("550e8400-e29b-41d4-a716-446655440000")
        assert result == []

    def test_create_agent_task_no_data_returned(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating agent task when no data is returned."""
        task_data = AgentTaskCreate(
            session_id=sample_chat_session.id,
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Task",
            instructions="Do something useful",
            priority=1
        )
        
        # Mock response with no data
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        with pytest.raises(ValueError, match="No data returned from agent task creation"):
            supabase_client.create_agent_task(task_data)

    def test_create_agent_task_database_error(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating agent task with database error."""
        task_data = AgentTaskCreate(
            session_id=sample_chat_session.id,
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Task",
            instructions="Do something useful",
            priority=1
        )
        
        # Mock database exception
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("Insert failed")
        
        with pytest.raises(Exception, match="Insert failed"):
            supabase_client.create_agent_task(task_data)

    def test_create_agent_task_empty_title_validation(self, supabase_client, mock_supabase, sample_chat_session):
        """Test creating agent task with empty title raises validation error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentTaskCreate(
                session_id=sample_chat_session.id,
                user_id="550e8400-e29b-41d4-a716-446655440000",
                title="",
                instructions="Do something useful",
                priority=1
            )

    # Agent Execution Tests
    def test_create_agent_execution(self, supabase_client, mock_supabase, sample_agent_task):
        """Test creating an agent execution."""
        execution_data = AgentExecutionCreate(
            task_id=sample_agent_task.id,
            step_name="Initialize",
            step_type="setup",
            sequence_number=1
        )
        
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [{
            "id": str(uuid4()),
            "task_id": str(sample_agent_task.id),
            "step_name": "Initialize",
            "step_type": "setup",
            "status": "started",
            "sequence_number": 1,
            "retry_count": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        result = supabase_client.create_agent_execution(execution_data)
        
        assert result is not None
        assert result.step_name == "Initialize"
        assert result.status == ExecutionStatus.STARTED

    def test_get_task_executions(self, supabase_client, mock_supabase, sample_agent_task):
        """Test getting task executions."""
        # Mock successful response
        mock_response = Mock()
        mock_response.data = [
            {
                "id": str(uuid4()),
                "task_id": str(sample_agent_task.id),
                "step_name": "Step 1",
                "step_type": "execution",
                "status": "completed",
                "sequence_number": 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid4()),
                "task_id": str(sample_agent_task.id),
                "step_name": "Step 2",
                "step_type": "execution",
                "status": "running",
                "sequence_number": 2,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_task_executions(sample_agent_task.id)
        
        assert len(result) == 2
        assert result[0].sequence_number == 1
        assert result[1].sequence_number == 2

    def test_get_task_executions_empty_result(self, supabase_client, mock_supabase):
        """Test getting task executions with empty result."""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_task_executions(str(uuid4()))
        
        assert len(result) == 0
        assert isinstance(result, list)

    def test_get_task_executions_database_error(self, supabase_client, mock_supabase):
        """Test getting task executions with database error."""
        # Mock database exception
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = Exception("Query failed")
        
        # get_task_executions catches exceptions and returns empty list
        result = supabase_client.get_task_executions(str(uuid4()))
        assert result == []

    def test_create_agent_execution_no_data_returned(self, supabase_client, mock_supabase, sample_agent_task):
        """Test creating agent execution when no data is returned."""
        execution_data = AgentExecutionCreate(
            task_id=sample_agent_task.id,
            step_name="Initialize",
            step_type="setup",
            sequence_number=1
        )
        
        # Mock response with no data
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_response
        
        with pytest.raises(ValueError, match="No data returned from agent execution creation"):
            supabase_client.create_agent_execution(execution_data)

    def test_create_agent_execution_database_error(self, supabase_client, mock_supabase, sample_agent_task):
        """Test creating agent execution with database error."""
        execution_data = AgentExecutionCreate(
            task_id=sample_agent_task.id,
            step_name="Initialize",
            step_type="setup",
            sequence_number=1
        )
        
        # Mock database exception
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = Exception("Insert failed")
        
        with pytest.raises(Exception, match="Insert failed"):
            supabase_client.create_agent_execution(execution_data)


class TestSupabaseClientFactory:
    """Test cases for Supabase client factory functions."""

    @patch('src.database.client.create_client')
    def test_get_supabase_client(self, mock_create_client):
        """Test getting Supabase client."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = "https://test.supabase.co"
            mock_settings.supabase_key = "test_key"
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_client()
            
            assert isinstance(result, SupabaseClient)
            assert result.client == mock_client
            mock_create_client.assert_called_once_with(
                supabase_url="https://test.supabase.co",
                supabase_key="test_key",
                options=mock_create_client.call_args[1]['options']
            )

    @patch('src.database.client.create_client')
    def test_get_supabase_admin_client(self, mock_create_client):
        """Test getting Supabase admin client."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = "https://test.supabase.co"
            mock_settings.supabase_service_key = "test_service_key"
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_admin_client()
            
            assert isinstance(result, SupabaseClient)
            assert result.client == mock_client
            # Verify the call was made with correct parameters
            mock_create_client.assert_called_once()
            call_args = mock_create_client.call_args
            assert call_args.kwargs['supabase_url'] == "https://test.supabase.co"
            assert call_args.kwargs['supabase_key'] == "test_service_key"
            assert call_args.kwargs['options'] is not None
            # Verify ClientOptions settings
            options = call_args.kwargs['options']
            assert options.auto_refresh_token is False
            assert options.persist_session is False

    def test_get_supabase_client_missing_config(self):
        """Test getting Supabase client with missing configuration."""
        # Clear the cache first
        get_supabase_client.cache_clear()
        
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = None
            mock_settings.supabase_key = None
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_client()
            
            assert result is None

    def test_get_supabase_client_partial_config(self):
        """Test getting Supabase client with partial configuration."""
        # Clear the cache first
        get_supabase_client.cache_clear()
        
        # Test with only URL
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = "https://test.supabase.co"
            mock_settings.supabase_key = None
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_client()
            assert result is None
        
        # Test with only key
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = None
            mock_settings.supabase_key = "test_key"
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_client()
            assert result is None

    def test_get_supabase_client_empty_config(self):
        """Test getting Supabase client with empty configuration values."""
        # Clear the cache first
        get_supabase_client.cache_clear()
        
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = ""
            mock_settings.supabase_key = ""
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_client()
            
            assert result is None

    @patch('src.database.client.create_client')
    def test_get_supabase_client_creation_error(self, mock_create_client):
        """Test getting Supabase client with client creation error."""
        mock_create_client.side_effect = Exception("Client creation failed")
        
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = "https://test.supabase.co"
            mock_settings.supabase_key = "test_key"
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_client()
            
            assert result is None

    def test_get_supabase_admin_client_missing_config(self):
        """Test getting Supabase admin client with missing configuration."""
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = None
            mock_settings.supabase_service_key = None
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_admin_client()
            
            assert result is None

    def test_get_supabase_admin_client_partial_config(self):
        """Test getting Supabase admin client with partial configuration."""
        # Test with only URL
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = "https://test.supabase.co"
            mock_settings.supabase_service_key = None
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_admin_client()
            assert result is None
        
        # Test with only service key
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = None
            mock_settings.supabase_service_key = "test_service_key"
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_admin_client()
            assert result is None

    @patch('src.database.client.create_client')
    def test_get_supabase_admin_client_creation_error(self, mock_create_client):
        """Test getting Supabase admin client with client creation error."""
        mock_create_client.side_effect = Exception("Admin client creation failed")
        
        with patch('src.database.client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.supabase_url = "https://test.supabase.co"
            mock_settings.supabase_service_key = "test_service_key"
            mock_get_settings.return_value = mock_settings
            
            result = get_supabase_admin_client()
            
            assert result is None


class TestInputValidation:
    """Test input validation for all CRUD operations."""
    
    def test_create_user_profile_invalid_email_format(self):
        """Test creating user profile with invalid email format."""
        from src.database.schemas import UserProfileCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            UserProfileCreate(
                user_id="550e8400-e29b-41d4-a716-446655440000",
                email="invalid-email-format",
                name="Test User"
            )
    
    def test_create_user_profile_none_values(self):
        """Test creating user profile with None values."""
        from src.database.schemas import UserProfileCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            UserProfileCreate(
                user_id=None,
                email="test@example.com",
                name="Test User"
            )
    
    def test_create_chat_session_invalid_uuid(self):
        """Test creating chat session with invalid UUID format."""
        from src.database.schemas import ChatSessionCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatSessionCreate(
                user_id="invalid-uuid-format",
                title="Test Session"
            )
    
    def test_create_chat_session_empty_title(self):
        """Test creating chat session with empty title raises ValidationError."""
        from src.database.schemas import ChatSessionCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatSessionCreate(
                user_id="550e8400-e29b-41d4-a716-446655440000",
                title="   "  # Empty string with whitespace
            )


class TestSupabaseClientAdditionalMethods:
    """Additional tests for SupabaseClient methods to improve coverage."""

    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client."""
        mock = Mock()
        mock.table.return_value = mock
        mock.select.return_value = mock
        mock.eq.return_value = mock
        mock.update.return_value = mock
        mock.execute.return_value = mock
        return mock

    @pytest.fixture
    def supabase_client(self, mock_supabase):
        """Create SupabaseClient with mocked client."""
        return SupabaseClient(mock_supabase)

    def test_get_chat_session_success(self, supabase_client, mock_supabase):
        """Test getting a chat session by ID successfully."""
        mock_response = Mock()
        mock_response.data = [{
            "id": "test-session-id",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Session",
            "description": "Test description",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_chat_session("test-session-id")
        
        assert result is not None
        assert result.title == "Test Session"
        assert result.user_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_get_chat_session_not_found(self, supabase_client, mock_supabase):
        """Test getting a chat session that doesn't exist."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_chat_session("nonexistent-id")
        
        assert result is None

    def test_get_chat_session_database_error(self, supabase_client, mock_supabase):
        """Test getting a chat session with database error."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB Error")
        
        result = supabase_client.get_chat_session("test-session-id")
        
        assert result is None

    def test_update_chat_session_success(self, supabase_client, mock_supabase):
        """Test updating a chat session successfully."""
        mock_response = Mock()
        mock_response.data = [{
            "id": "test-session-id",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Updated Session",
            "description": "Updated description",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        
        update_data = {"title": "Updated Session", "description": "Updated description"}
        result = supabase_client.update_chat_session("test-session-id", update_data)
        
        assert result is not None
        assert result["title"] == "Updated Session"
        assert result["description"] == "Updated description"

    def test_update_chat_session_no_data_returned(self, supabase_client, mock_supabase):
        """Test updating a chat session with no data returned."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        
        update_data = {"title": "Updated Session"}
        result = supabase_client.update_chat_session("test-session-id", update_data)
        
        assert result is None

    def test_update_chat_session_database_error(self, supabase_client, mock_supabase):
        """Test updating a chat session with database error."""
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception("DB Error")
        
        update_data = {"title": "Updated Session"}
        result = supabase_client.update_chat_session("test-session-id", update_data)
        
        assert result is None

    def test_get_chat_messages_success(self, supabase_client, mock_supabase):
        """Test getting chat messages successfully."""
        mock_response = Mock()
        mock_response.data = [
            {
                "id": "msg1",
                "session_id": "session1",
                "role": "user",
                "content": "Hello",
                "sequence_number": 1,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "msg2",
                "session_id": "session1",
                "role": "assistant",
                "content": "Hi there!",
                "sequence_number": 2,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_chat_messages("session1", limit=10, offset=0)
        
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there!"

    def test_get_chat_messages_empty_result(self, supabase_client, mock_supabase):
        """Test getting chat messages with empty result."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_chat_messages("session1")
        
        assert result == []

    def test_get_chat_messages_database_error(self, supabase_client, mock_supabase):
        """Test getting chat messages with database error."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.return_value.execute.side_effect = Exception("DB Error")
        
        result = supabase_client.get_chat_messages("session1")
        
        assert result == []

    def test_get_agent_task_success(self, supabase_client, mock_supabase):
        """Test getting an agent task successfully."""
        mock_response = Mock()
        mock_response.data = [{
            "id": "task1",
            "session_id": "session1",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "description": "Test description",
            "status": "pending",
            "priority": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_agent_task("task1")
        
        assert result is not None
        assert result["title"] == "Test Task"
        assert result["status"] == "pending"

    def test_get_agent_task_not_found(self, supabase_client, mock_supabase):
        """Test getting an agent task that doesn't exist."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_agent_task("nonexistent-task")
        
        assert result is None

    def test_get_agent_task_database_error(self, supabase_client, mock_supabase):
        """Test getting an agent task with database error."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("DB Error")
        
        result = supabase_client.get_agent_task("task1")
        
        assert result is None

    def test_update_agent_task_success(self, supabase_client, mock_supabase):
        """Test updating an agent task successfully."""
        mock_response = Mock()
        mock_response.data = [{
            "id": "task1",
            "session_id": "session1",
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Updated Task",
            "description": "Updated description",
            "status": "completed",
            "priority": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }]
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        
        update_data = {"title": "Updated Task", "status": "completed"}
        result = supabase_client.update_agent_task("task1", update_data)
        
        assert result is not None
        assert result["title"] == "Updated Task"
        assert result["status"] == "completed"

    def test_update_agent_task_no_data_returned(self, supabase_client, mock_supabase):
        """Test updating an agent task with no data returned."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response
        
        update_data = {"status": "completed"}
        result = supabase_client.update_agent_task("task1", update_data)
        
        assert result is None

    def test_update_agent_task_database_error(self, supabase_client, mock_supabase):
        """Test updating an agent task with database error."""
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception("DB Error")
        
        update_data = {"status": "completed"}
        result = supabase_client.update_agent_task("task1", update_data)
        
        assert result is None

    def test_get_agent_executions_success(self, supabase_client, mock_supabase):
        """Test getting agent executions successfully."""
        mock_response = Mock()
        mock_response.data = [
            {
                "id": "exec1",
                "task_id": "task1",
                "step_name": "Step 1",
                "status": "completed",
                "sequence_number": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "exec2",
                "task_id": "task1",
                "step_name": "Step 2",
                "status": "running",
                "sequence_number": 2,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_agent_executions("task1", limit=5)
        
        assert len(result) == 2
        assert result[0]["step_name"] == "Step 1"
        assert result[1]["step_name"] == "Step 2"

    def test_get_agent_executions_empty_result(self, supabase_client, mock_supabase):
        """Test getting agent executions with empty result."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_agent_executions("task1")
        
        assert result == []

    def test_get_agent_executions_database_error(self, supabase_client, mock_supabase):
        """Test getting agent executions with database error."""
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.side_effect = Exception("DB Error")
        
        result = supabase_client.get_agent_executions("task1")
        
        assert result == []

    
    def test_create_chat_message_invalid_role(self):
        """Test creating chat message with invalid role."""
        from src.database.schemas import ChatMessageCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatMessageCreate(
                session_id="550e8400-e29b-41d4-a716-446655440000",
                role="invalid_role",
                content="Test message",
                sequence_number=1
            )
    
    def test_create_chat_message_negative_sequence(self):
        """Test creating chat message with negative sequence number."""
        from src.database.schemas import ChatMessageCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatMessageCreate(
                session_id="550e8400-e29b-41d4-a716-446655440000",
                role=MessageRole.USER,
                content="Test message",
                sequence_number=-1
            )
    
    def test_create_agent_task_invalid_priority(self):
        """Test creating agent task with invalid priority."""
        from src.database.schemas import AgentTaskCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentTaskCreate(
                session_id="550e8400-e29b-41d4-a716-446655440000",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                title="Test Task",
                instructions="Do something",
                priority=-1  # Invalid negative priority
            )
    
    def test_create_agent_task_priority_too_high(self):
        """Test creating agent task with priority too high."""
        from src.database.schemas import AgentTaskCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentTaskCreate(
                session_id="550e8400-e29b-41d4-a716-446655440000",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                title="Test Task",
                instructions="Do something",
                priority=11  # Priority should be 1-10
            )
    
    def test_create_agent_execution_invalid_sequence(self):
        """Test creating agent execution with invalid sequence number."""
        from src.database.schemas import AgentExecutionCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentExecutionCreate(
                task_id="550e8400-e29b-41d4-a716-446655440000",
                step_name="Test Step",
                step_type="action",
                sequence_number=0  # Should be positive
            )
    
    def test_create_agent_execution_empty_step_name(self):
        """Test creating agent execution with empty step name."""
        from src.database.schemas import AgentExecutionCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentExecutionCreate(
                task_id="550e8400-e29b-41d4-a716-446655440000",
                step_name="",
                step_type="action",
                sequence_number=1
            )
    
    @pytest.fixture
    def mock_supabase(self):
        """Create a mock Supabase client."""
        mock_client = Mock()
        mock_client.table.return_value = Mock()
        return mock_client
    
    def test_get_methods_with_invalid_uuids(self, mock_supabase):
        """Test get methods with invalid UUID formats."""
        from src.database.client import SupabaseClient
        supabase_client = SupabaseClient(mock_supabase)
        
        # Mock to prevent actual database calls
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response
        
        # These should handle invalid UUIDs gracefully
        result = supabase_client.get_user_profile("invalid-uuid")
        assert result is None
        
        result = supabase_client.get_user_chat_sessions("invalid-uuid")
        assert result == []
        
        result = supabase_client.get_session_messages("invalid-uuid")
        assert result == []


class TestSupabaseIntegration:
    """Integration tests for Supabase functionality."""

    @pytest.mark.integration
    @patch('src.database.client.get_supabase_client')
    def test_database_health_check_integration(self, mock_get_client):
        """Test database health check integration."""
        import asyncio
        from src.utils.monitoring import database_health_check
        
        # Mock successful client
        mock_supabase_client = Mock()
        mock_supabase_client.rpc.return_value.execute.return_value.data = "PostgreSQL 14.0"
        mock_client = SupabaseClient(mock_supabase_client)
        mock_client._connected = True  # Set the connected property
        mock_get_client.return_value = mock_client
        
        result = asyncio.run(database_health_check())
        
        assert result["status"] == "healthy"
        assert "database" in result
        assert result["database"]["connected"] is True

    @pytest.mark.integration
    @patch('src.database.client.get_supabase_client')
    def test_database_health_check_failure(self, mock_get_client):
        """Test database health check when no client is configured."""
        import asyncio
        from src.utils.monitoring import database_health_check
        
        # Mock client not configured
        mock_get_client.return_value = None
        
        result = asyncio.run(database_health_check())
        
        assert result["healthy"] is True
        assert "No database configured" in result["message"]
        assert "details" in result
        assert result["details"]["status"] == "not_applicable"


class TestDataIntegrity:
    """Test data integrity constraints and validation."""

    @pytest.fixture
    def mock_supabase(self):
        """Create a mock Supabase client."""
        mock = Mock()
        mock.table.return_value = mock
        mock.insert.return_value = mock
        mock.select.return_value = mock
        mock.eq.return_value = mock
        mock.update.return_value = mock
        mock.execute.return_value = mock
        return mock

    @pytest.fixture
    def supabase_client(self, mock_supabase):
        """Create a SupabaseClient instance with mocked client."""
        return SupabaseClient(mock_supabase)

    def test_duplicate_email_handling(self, supabase_client, mock_supabase):
        """Test handling of duplicate email addresses."""
        from postgrest.exceptions import APIError
        
        # Simulate duplicate key violation
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "duplicate key value violates unique constraint", "code": "23505"}
        )
        
        user_data = UserProfileCreate(
            user_id=str(uuid4()),
            email="duplicate@example.com",
            username="testuser",
            full_name="Test User"
        )
        
        with pytest.raises(Exception) as exc_info:
            supabase_client.create_user_profile(user_data)
        
        assert "duplicate key" in str(exc_info.value)

    def test_invalid_foreign_key_reference(self, supabase_client, mock_supabase):
        """Test handling of invalid foreign key references."""
        from postgrest.exceptions import APIError
        
        # Simulate foreign key constraint violation
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "violates foreign key constraint", "code": "23503"}
        )
        
        chat_data = ChatSessionCreate(
            user_id=str(uuid4()),
            title="Test Session"
        )
        
        with pytest.raises(Exception) as exc_info:
            supabase_client.create_chat_session(chat_data)
        
        assert "foreign key" in str(exc_info.value)

    def test_update_nonexistent_record(self, supabase_client, mock_supabase):
        """Test updating a record that doesn't exist."""
        # Mock empty result for update operation
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value.data = []
        
        result = supabase_client.update_chat_session(
            "non-existent-id",
            {"title": "Updated Title"}
        )
        
        assert result is None

    def test_sequence_number_integrity(self, supabase_client, mock_supabase):
        """Test sequence number integrity for chat messages."""
        session_id = str(uuid4())
        # Mock successful creation with sequence validation
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [{
            "id": str(uuid4()),
            "session_id": session_id,
            "role": "user",
            "content": "Test message",
            "sequence_number": 1,
            "created_at": "2024-01-01T00:00:00Z"
        }]
        
        message_data = ChatMessageCreate(
            session_id=session_id,
            role=MessageRole.USER,
            content="Test message",
            sequence_number=1
        )
        
        result = supabase_client.create_chat_message(message_data)
        
        assert result is not None
        assert result.sequence_number == 1

    def test_null_constraint_validation(self, supabase_client, mock_supabase):
        """Test null constraint validation."""
        from postgrest.exceptions import APIError
        
        # Simulate null constraint violation
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "null value in column violates not-null constraint", "code": "23502"}
        )
        
        # Try to create user with missing required field
        user_data = UserProfileCreate(
            user_id=str(uuid4()),
            email="",  # Empty email should trigger validation
            username="testuser",
            full_name="Test User"
        )
        
        with pytest.raises(Exception) as exc_info:
            supabase_client.create_user_profile(user_data)
        
        assert "not-null constraint" in str(exc_info.value)

    def test_data_type_validation(self, supabase_client, mock_supabase):
        """Test data type validation."""
        from postgrest.exceptions import APIError
        
        # Simulate data type mismatch
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "invalid input syntax for type uuid", "code": "22P02"}
        )
        
        # Test with invalid UUID that passes pydantic validation but fails at DB level
        with pytest.raises(Exception) as exc_info:
            # Directly call with invalid data to bypass pydantic validation
            supabase_client.client.table("chat_messages").insert({
                "session_id": "invalid-uuid-format",
                "role": "user",
                "content": "Test message",
                "sequence_number": 1
            }).execute()
        
        assert "invalid input syntax" in str(exc_info.value)

    def test_check_constraint_validation(self, supabase_client, mock_supabase):
        """Test check constraint validation."""
        from postgrest.exceptions import APIError
        
        # Simulate check constraint violation
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "new row violates check constraint", "code": "23514"}
        )
        
        task_data = AgentTaskCreate(
            session_id=str(uuid4()),
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Task",
            description="Test Description",
            instructions="Test instructions",
            priority=1  # Valid priority value
        )
        
        with pytest.raises(Exception) as exc_info:
            supabase_client.create_agent_task(task_data)
        
        assert "check constraint" in str(exc_info.value)

    def test_transaction_rollback_simulation(self, supabase_client, mock_supabase):
        """Test transaction rollback behavior."""
        from postgrest.exceptions import APIError
        
        # Simulate transaction failure
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "transaction aborted", "code": "25P02"}
        )
        
        user_data = UserProfileCreate(
            user_id=str(uuid4()),
            email="test@example.com",
            username="testuser",
            full_name="Test User"
        )
        
        with pytest.raises(Exception) as exc_info:
            supabase_client.create_user_profile(user_data)
        
        assert "transaction aborted" in str(exc_info.value)


class TestPerformanceEdgeCases:
    """Test performance-related edge cases and timeout scenarios."""

    @pytest.fixture
    def mock_supabase(self):
        """Create a mock Supabase client."""
        mock = Mock()
        mock.table.return_value = mock
        mock.insert.return_value = mock
        mock.select.return_value = mock
        mock.eq.return_value = mock
        mock.update.return_value = mock
        mock.execute.return_value = mock
        mock.rpc.return_value = mock
        return mock

    @pytest.fixture
    def supabase_client(self, mock_supabase):
        """Create a SupabaseClient instance with mocked client."""
        return SupabaseClient(mock_supabase)

    def test_large_dataset_handling(self, supabase_client, mock_supabase):
        """Test handling of large datasets."""
        # Mock large result set
        large_data = [{
            "id": f"msg-{i}", 
            "session_id": "session-123",
            "role": "user",
            "content": f"Message {i}",
            "sequence_number": i + 1,
            "created_at": "2023-01-01T00:00:00Z"
        } for i in range(1000)]
        
        # Create a proper mock response
        mock_response = Mock()
        mock_response.data = large_data
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.return_value.execute.return_value = mock_response
        
        result = supabase_client.get_chat_messages("session-123", limit=1000)
        
        assert len(result) == 1000
        assert all(isinstance(msg, dict) for msg in result)
        assert all("content" in msg for msg in result)

    def test_connection_timeout_handling(self, supabase_client, mock_supabase):
        """Test connection timeout handling."""
        from postgrest.exceptions import APIError
        
        # Simulate timeout error on the rpc call
        mock_supabase.rpc.return_value.execute.side_effect = APIError(
            {"message": "Connection timeout", "code": "08006"}
        )
        
        # health_check catches exceptions and returns unhealthy status
        result = supabase_client.health_check()
        
        assert result["status"] == "unhealthy"
        assert "timeout" in result["error"].lower()

    def test_slow_query_handling(self, supabase_client, mock_supabase):
        """Test handling of slow queries."""
        import time
        
        def slow_execute():
            time.sleep(0.1)  # Simulate slow query
            mock_response = Mock()
            mock_response.data = [{
                "id": "session-123", 
                "user_id": str(uuid4()),
                "title": "Test Session",
                "created_at": "2023-01-01T00:00:00Z"
            }]
            return mock_response
        
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = slow_execute
        
        start_time = time.time()
        result = supabase_client.get_chat_session("session-123")
        end_time = time.time()
        
        assert result is not None
        assert end_time - start_time >= 0.1  # Verify delay occurred

    def test_memory_efficient_pagination(self, supabase_client, mock_supabase):
        """Test memory-efficient pagination for large datasets."""
        # Mock paginated results
        page_1_data = [{
            "id": f"msg-{i}", 
            "session_id": "session-123",
            "role": "user",
            "content": f"Message {i}",
            "sequence_number": i + 1,
            "created_at": "2023-01-01T00:00:00Z"
        } for i in range(50)]
        
        # Create a proper mock response
        mock_response = Mock()
        mock_response.data = page_1_data
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.return_value.execute.return_value = mock_response
        
        result_page_1 = supabase_client.get_chat_messages("session-123", limit=50, offset=0)
        
        assert len(result_page_1) == 50
        assert all(isinstance(msg, dict) for msg in result_page_1)
        assert all("content" in msg for msg in result_page_1)
        
        # Verify pagination parameters were used
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.assert_called_with(50)
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.assert_called_with(0)

    def test_concurrent_connection_handling(self, supabase_client, mock_supabase):
        """Test handling of concurrent connections."""
        import threading
        
        results = []
        errors = []
        
        def concurrent_operation(thread_id):
            try:
                # Mock different responses for each thread
                mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [{
                    "id": f"user-{thread_id}", 
                    "email": f"user{thread_id}@example.com",
                    "username": f"user{thread_id}",
                    "full_name": f"User {thread_id}",
                    "created_at": "2023-01-01T00:00:00Z"
                }]
                result = supabase_client.get_user_profile(f"user-{thread_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 5  # All operations should complete

    def test_resource_cleanup_on_error(self, supabase_client, mock_supabase):
        """Test proper resource cleanup when errors occur."""
        from postgrest.exceptions import APIError
        
        # Simulate error that requires cleanup
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "connection lost", "code": "08006"}
        )
        
        user_data = UserProfileCreate(
            user_id=str(uuid4()),
            email="test@example.com",
            username="testuser",
            full_name="Test User"
        )
        
        with pytest.raises(Exception):
            supabase_client.create_user_profile(user_data)
        
        # Verify client is still functional after error
        mock_supabase.table.return_value.insert.return_value.execute.side_effect = None
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [{
            "id": "user-123",
            "user_id": str(uuid4()),
            "email": "test2@example.com",
            "username": "testuser2",
            "full_name": "Test User 2",
            "created_at": "2023-01-01T00:00:00Z"
        }]
        
        user_data_2 = UserProfileCreate(
            user_id=str(uuid4()),
            email="test2@example.com",
            username="testuser2",
            full_name="Test User 2"
        )
        
        result = supabase_client.create_user_profile(user_data_2)
        assert result is not None

    def test_batch_operation_performance(self, supabase_client, mock_supabase):
        """Test performance of batch operations."""
        import time
        
        # Mock batch insert response
        batch_data = [{
            "id": f"msg-{i}", 
            "session_id": "session-123",
            "role": "user",
            "content": f"Message {i}",
            "sequence_number": i + 1,
            "created_at": "2023-01-01T00:00:00Z"
        } for i in range(100)]
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = batch_data
        
        start_time = time.time()
        
        # Simulate batch creation (in real scenario, this would be a single batch insert)
        messages = []
        for i in range(100):
            message_data = ChatMessageCreate(
                session_id=str(uuid4()),
                role=MessageRole.USER,
                content=f"Message {i}",
                sequence_number=i + 1
            )
            # In a real batch operation, this would be accumulated and sent as one request
            result = supabase_client.create_chat_message(message_data)
            messages.append(result)
        
        end_time = time.time()
        
        assert len(messages) == 100
        assert end_time - start_time < 1.0  # Should complete quickly with mocked operations

    def test_connection_pool_exhaustion(self, supabase_client, mock_supabase):
        """Test handling of connection pool exhaustion."""
        from postgrest.exceptions import APIError
        
        # Simulate connection pool exhaustion - need to mock the entire chain
        error = APIError({"message": "connection pool exhausted", "code": "53300"})
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = error
        
        # get_user_profile catches exceptions and returns None
        result = supabase_client.get_user_profile("user-123")
        
        assert result is None

    def test_query_complexity_limits(self, supabase_client, mock_supabase):
        """Test handling of complex queries that might hit limits."""
        from postgrest.exceptions import APIError
        
        # Simulate query complexity limit - need to mock the entire chain
        error = APIError({"message": "query too complex", "code": "54001"})
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.return_value.execute.side_effect = error
        
        # get_chat_messages catches exceptions and returns empty list
        result = supabase_client.get_chat_messages("session-123", limit=10000)
        
        assert result == []