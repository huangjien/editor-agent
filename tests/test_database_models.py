"""Tests for Supabase database models and schemas."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4
from pydantic import ValidationError

from src.database.models import (
    UserProfile, ChatSession, ChatMessage, AgentTask, AgentExecution, FileOperation,
    TaskStatus, MessageRole, ExecutionStatus, FileOperationType
)
from src.database.schemas import (
    UserProfileCreate, UserProfileUpdate,
    ChatSessionCreate, ChatMessageCreate, AgentTaskCreate, AgentTaskUpdate,
    FileOperationCreate, FileOperationUpdate
)


class TestEnums:
    """Test cases for database enums."""

    def test_task_status_enum(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"

    def test_message_role_enum(self):
        """Test MessageRole enum values."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"

    def test_execution_status_enum(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.STARTED == "started"
        assert ExecutionStatus.RUNNING == "running"
        assert ExecutionStatus.COMPLETED == "completed"
        assert ExecutionStatus.FAILED == "failed"
        assert ExecutionStatus.TIMEOUT == "timeout"

    def test_file_operation_type_enum(self):
        """Test FileOperationType enum values."""
        assert FileOperationType.CREATE == "create"
        assert FileOperationType.READ == "read"
        assert FileOperationType.UPDATE == "update"
        assert FileOperationType.DELETE == "delete"
        assert FileOperationType.RENAME == "rename"
        assert FileOperationType.MOVE == "move"


class TestUserProfile:
    """Test cases for UserProfile model."""

    def test_user_profile_creation(self):
        """Test creating a valid UserProfile."""
        profile = UserProfile(
            id=str(uuid4()),
            user_id="test_user_123",
            email="test@example.com",
            name="Test User",
            preferences={"theme": "dark", "language": "en"},
            metadata={"source": "registration"},
            is_active=True,
            last_login=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        assert profile.user_id == "test_user_123"
        assert profile.email == "test@example.com"
        assert profile.name == "Test User"
        assert profile.preferences["theme"] == "dark"
        assert profile.is_active is True

    def test_user_profile_defaults(self):
        """Test UserProfile default values."""
        profile = UserProfile(
            id=str(uuid4()),
            user_id="test_user_123"
        )
        
        assert profile.preferences == {}
        assert profile.metadata == {}
        assert profile.is_active is True
        assert profile.email is None
        assert profile.name is None


class TestUserProfileSchemas:
    """Test cases for UserProfile schemas."""

    def test_user_profile_create_valid(self):
        """Test creating a valid UserProfileCreate schema."""
        test_uuid = str(uuid4())
        create_data = UserProfileCreate(
            user_id=test_uuid,
            email="test@example.com",
            name="Test User",
            preferences={"theme": "dark"}
        )
        
        assert create_data.user_id == test_uuid
        assert create_data.email == "test@example.com"
        assert create_data.name == "Test User"
        assert create_data.preferences["theme"] == "dark"

    def test_user_profile_create_minimal(self):
        """Test creating UserProfileCreate with minimal data."""
        test_uuid = str(uuid4())
        create_data = UserProfileCreate(user_id=test_uuid)
        
        assert create_data.user_id == test_uuid
        assert create_data.email is None
        assert create_data.name is None
        assert create_data.preferences == {}

    def test_user_profile_create_invalid_email(self):
        """Test UserProfileCreate with invalid email."""
        test_uuid = str(uuid4())
        with pytest.raises(ValidationError):
            UserProfileCreate(
                user_id=test_uuid,
                email="invalid-email"
            )

    def test_user_profile_update(self):
        """Test UserProfileUpdate schema."""
        update_data = UserProfileUpdate(
            name="Updated Name",
            preferences={"theme": "light"}
        )
        
        assert update_data.name == "Updated Name"
        assert update_data.preferences["theme"] == "light"
        assert update_data.email is None  # Not updated


class TestChatSession:
    """Test cases for ChatSession model."""

    def test_chat_session_creation(self):
        """Test creating a valid ChatSession."""
        session = ChatSession(
            id=str(uuid4()),
            user_id="test_user_123",
            title="Test Session",
            description="A test chat session",
            context={"mode": "coding"},
            settings={"model": "gpt-4"},
            is_active=True,
            message_count=5,
            tags=["coding", "python"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        assert session.user_id == "test_user_123"
        assert session.title == "Test Session"
        assert session.message_count == 5
        assert "coding" in session.tags

    def test_chat_session_defaults(self):
        """Test ChatSession default values."""
        session = ChatSession(
            id=str(uuid4()),
            user_id="test_user_123"
        )
        
        assert session.context == {}
        assert session.settings == {}
        assert session.is_active is True
        assert session.message_count == 0
        assert session.tags == []


class TestChatSessionSchemas:
    """Test cases for ChatSession schemas."""

    def test_chat_session_create_valid(self):
        """Test creating a valid ChatSessionCreate schema."""
        create_data = ChatSessionCreate(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Test Session",
            description="A test session",
            context={"mode": "coding"},
            tags=["python", "ai"]
        )
        
        assert create_data.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert create_data.title == "Test Session"
        assert create_data.context["mode"] == "coding"
        assert "python" in create_data.tags

    def test_chat_session_create_minimal(self):
        """Test creating ChatSessionCreate with minimal data."""
        create_data = ChatSessionCreate(user_id="550e8400-e29b-41d4-a716-446655440000")
        
        assert create_data.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert create_data.title is None
        assert create_data.context == {}
        assert create_data.tags == []


class TestChatMessage:
    """Test cases for ChatMessage model."""

    def test_chat_message_creation(self):
        """Test creating a valid ChatMessage."""
        session_id = str(uuid4())
        message = ChatMessage(
            id=str(uuid4()),
            session_id=session_id,
            role=MessageRole.USER,
            content="Hello, world!",
            metadata={"source": "web"},
            tool_calls=[{"name": "search", "args": {"query": "test"}}],
            tool_results=[{"result": "success"}],
            sequence_number=1,
            tokens_used=50,
            model_used="gpt-4",
            processing_time=1.5,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        assert message.session_id == session_id
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.sequence_number == 1
        assert message.tokens_used == 50

    def test_chat_message_defaults(self):
        """Test ChatMessage default values."""
        message = ChatMessage(
            id=str(uuid4()),
            session_id=str(uuid4()),
            role=MessageRole.USER,
            content="Test message",
            sequence_number=1
        )
        
        assert message.metadata == {}
        assert message.tool_calls == []
        assert message.tool_results == []
        assert message.parent_message_id is None
        assert message.tokens_used is None


class TestChatMessageSchemas:
    """Test cases for ChatMessage schemas."""

    def test_chat_message_create_valid(self):
        """Test creating a valid ChatMessageCreate schema."""
        session_id = str(uuid4())
        create_data = ChatMessageCreate(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content="Hello! How can I help you?",
            sequence_number=2,
            metadata={"confidence": 0.95}
        )
        
        assert create_data.session_id == session_id
        assert create_data.role == MessageRole.ASSISTANT
        assert create_data.content == "Hello! How can I help you?"
        assert create_data.sequence_number == 2

    def test_chat_message_create_invalid_role(self):
        """Test ChatMessageCreate with invalid role."""
        with pytest.raises(ValidationError):
            ChatMessageCreate(
                session_id=str(uuid4()),
                role="invalid_role",  # type: ignore
                content="Test",
                sequence_number=1
            )

    def test_chat_message_create_negative_sequence(self):
        """Test ChatMessageCreate with negative sequence number."""
        with pytest.raises(ValidationError):
            ChatMessageCreate(
                session_id=str(uuid4()),
                role=MessageRole.USER,
                content="Test",
                sequence_number=-1
            )


class TestAgentTask:
    """Test cases for AgentTask model."""

    def test_agent_task_creation(self):
        """Test creating a valid AgentTask."""
        session_id = str(uuid4())
        task = AgentTask(
            id=str(uuid4()),
            session_id=session_id,
            user_id="test_user_123",
            title="Code Review",
            description="Review the Python code",
            instructions="Check for bugs and improvements",
            status=TaskStatus.IN_PROGRESS,
            priority=2,
            context={"language": "python"},
            requirements=["pylint", "mypy"],
            deliverables=["review_report.md"],
            tags=["code-review", "python"],
            estimated_duration=60,
            progress=25,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        assert task.session_id == session_id
        assert task.title == "Code Review"
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.priority == 2
        assert task.progress == 25
        assert "python" in task.tags

    def test_agent_task_defaults(self):
        """Test AgentTask default values."""
        task = AgentTask(
            id=str(uuid4()),
            user_id="test_user_123",
            title="Test Task",
            instructions="Do something"
        )
        
        assert task.status == TaskStatus.PENDING
        assert task.priority == 1
        assert task.context == {}
        assert task.requirements == []
        assert task.deliverables == []
        assert task.tags == []
        assert task.progress == 0


class TestAgentTaskSchemas:
    """Test cases for AgentTask schemas."""

    def test_agent_task_create_valid(self):
        """Test creating a valid AgentTaskCreate schema."""
        session_id = str(uuid4())
        create_data = AgentTaskCreate(
            session_id=session_id,
            user_id="test_user_123",
            title="Test Task",
            description="A test task",
            instructions="Follow these steps",
            priority=3,
            requirements=["requirement1"],
            tags=["test"]
        )
        
        assert create_data.session_id == session_id
        assert create_data.user_id == "test_user_123"
        assert create_data.title == "Test Task"
        assert create_data.priority == 3
        assert "test" in create_data.tags

    def test_agent_task_create_invalid_priority(self):
        """Test AgentTaskCreate with invalid priority."""
        with pytest.raises(ValidationError):
            AgentTaskCreate(
                user_id="test_user_123",
                title="Test Task",
                instructions="Test",
                priority=0  # Invalid: must be >= 1
            )

        with pytest.raises(ValidationError):
            AgentTaskCreate(
                user_id="test_user_123",
                title="Test Task",
                instructions="Test",
                priority=6  # Invalid: must be <= 5
            )

    def test_agent_task_update_progress(self):
        """Test AgentTaskUpdate with progress validation."""
        update_data = AgentTaskUpdate(progress=75)
        assert update_data.progress == 75
        
        with pytest.raises(ValidationError):
            AgentTaskUpdate(progress=-1)  # Invalid: must be >= 0
        
        with pytest.raises(ValidationError):
            AgentTaskUpdate(progress=101)  # Invalid: must be <= 100


class TestAgentExecution:
    """Test cases for AgentExecution model."""

    def test_agent_execution_creation(self):
        """Test creating a valid AgentExecution."""
        task_id = str(uuid4())
        execution = AgentExecution(
            id=str(uuid4()),
            task_id=task_id,
            step_name="Initialize",
            step_type="setup",
            status=ExecutionStatus.COMPLETED,
            input_data={"config": "test"},
            output_data={"result": "success"},
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration=2.5,
            sequence_number=1,
            retry_count=0,
            logs=["Starting", "Processing", "Complete"],
            metrics={"cpu_usage": 0.5},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        assert execution.task_id == task_id
        assert execution.step_name == "Initialize"
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.sequence_number == 1
        assert execution.duration == 2.5
        assert len(execution.logs) == 3

    def test_agent_execution_defaults(self):
        """Test AgentExecution default values."""
        from datetime import datetime
        execution = AgentExecution(
            id=str(uuid4()),
            task_id=str(uuid4()),
            step_name="Test Step",
            step_type="test",
            sequence_number=1,
            started_at=datetime.now()
        )
        
        assert execution.status == ExecutionStatus.STARTED
        assert execution.input_data is None
        assert execution.output_data is None
        assert execution.retry_count == 0
        assert execution.logs == []
        assert execution.metrics == {}


class TestFileOperation:
    """Test cases for FileOperation model."""

    def test_file_operation_creation(self):
        """Test creating a valid FileOperation."""
        session_id = str(uuid4())
        task_id = str(uuid4())
        operation = FileOperation(
            id=str(uuid4()),
            session_id=session_id,
            task_id=task_id,
            user_id="test_user_123",
            operation_type=FileOperationType.UPDATE,
            file_path="/path/to/file.py",
            old_path="/old/path/to/file.py",
            file_size=1024,
            file_hash="abc123",
            content_preview="def hello():",
            metadata={"encoding": "utf-8"},
            success=True,
            backup_path="/backup/file.py",
            changes_summary="Added new function",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        assert operation.session_id == session_id
        assert operation.task_id == task_id
        assert operation.operation_type == FileOperationType.UPDATE
        assert operation.file_path == "/path/to/file.py"
        assert operation.success is True

    def test_file_operation_defaults(self):
        """Test FileOperation default values."""
        operation = FileOperation(
            id=str(uuid4()),
            user_id="test_user_123",
            operation_type=FileOperationType.CREATE,
            file_path="/new/file.py"
        )
        
        assert operation.session_id is None
        assert operation.task_id is None
        assert operation.metadata == {}
        assert operation.success is True
        assert operation.error_message is None


class TestFileOperationSchemas:
    """Test cases for FileOperation schemas."""

    def test_file_operation_create_valid(self):
        """Test creating a valid FileOperationCreate schema."""
        create_data = FileOperationCreate(
            user_id="test_user_123",
            operation_type=FileOperationType.CREATE,
            file_path="/new/file.py",
            file_size=512,
            content_preview="# New file",
            metadata={"language": "python"}
        )
        
        assert create_data.user_id == "test_user_123"
        assert create_data.operation_type == FileOperationType.CREATE
        assert create_data.file_path == "/new/file.py"
        assert create_data.file_size == 512

    def test_file_operation_create_invalid_operation_type(self):
        """Test FileOperationCreate with invalid operation type."""
        with pytest.raises(ValidationError):
            FileOperationCreate(
                user_id="test_user_123",
                operation_type="invalid_type",  # type: ignore
                file_path="/file.py"
            )

    def test_file_operation_update(self):
        """Test FileOperationUpdate schema."""
        update_data = FileOperationUpdate(
            success=False,
            error_message="Permission denied",
            changes_summary="Failed to update file"
        )
        
        assert update_data.success is False
        assert update_data.error_message == "Permission denied"
        assert update_data.changes_summary == "Failed to update file"


class TestSchemaValidation:
    """Test cases for schema validation rules."""

    def test_email_validation(self):
        """Test email validation in schemas."""
        # Valid emails
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ]
        
        test_uuid = str(uuid4())
        for email in valid_emails:
            profile = UserProfileCreate(user_id=test_uuid, email=email)
            assert profile.email == email
        
        # Invalid emails
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "user@",
            "user space@example.com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                UserProfileCreate(user_id=test_uuid, email=email)

    def test_uuid_validation(self):
        """Test UUID validation in schemas."""
        valid_uuid = uuid4()
        
        # Valid UUID
        message = ChatMessageCreate(
            session_id=str(valid_uuid),
            role=MessageRole.USER,
            content="Test",
            sequence_number=1
        )
        assert message.session_id == str(valid_uuid)
        
        # Invalid UUID (string)
        with pytest.raises(ValidationError):
            ChatMessageCreate(
                session_id="not-a-uuid",  # type: ignore
                role=MessageRole.USER,
                content="Test",
                sequence_number=1
            )

    def test_positive_integer_validation(self):
        """Test positive integer validation."""
        # Valid positive integers
        for seq_num in [1, 10, 100]:
            message = ChatMessageCreate(
                session_id=str(uuid4()),
                role=MessageRole.USER,
                content="Test",
                sequence_number=seq_num
            )
            assert message.sequence_number == seq_num
        
        # Invalid: zero and negative
        for seq_num in [0, -1, -10]:
            with pytest.raises(ValidationError):
                ChatMessageCreate(
                    session_id=str(uuid4()),
                    role=MessageRole.USER,
                    content="Test",
                    sequence_number=seq_num
                )

    def test_range_validation(self):
        """Test range validation for priority and progress fields."""
        # Valid priority range (1-5)
        for priority in [1, 2, 3, 4, 5]:
            task = AgentTaskCreate(
                user_id="test",
                title="Test",
                instructions="Test",
                priority=priority
            )
            assert task.priority == priority
        
        # Invalid priority range
        for priority in [0, 6, -1, 10]:
            with pytest.raises(ValidationError):
                AgentTaskCreate(
                    user_id="test",
                    title="Test",
                    instructions="Test",
                    priority=priority
                )
        
        # Valid progress range (0-100)
        for progress in [0, 25, 50, 75, 100]:
            update = AgentTaskUpdate(progress=progress)
            assert update.progress == progress
        
        # Invalid progress range
        for progress in [-1, 101, -10, 150]:
            with pytest.raises(ValidationError):
                AgentTaskUpdate(progress=progress)