"""Comprehensive unit tests for database schemas."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.database.schemas import (
    BaseSchema,
    UserProfileCreate,
    UserProfileUpdate,
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatMessageCreate,
    ChatMessageUpdate,
    AgentTaskCreate,
    AgentTaskUpdate,
    AgentExecutionCreate,
    AgentExecutionUpdate,
    FileOperationCreate,
    FileOperationUpdate,
)
from src.database.models import (
    TaskStatus,
    MessageRole,
    ExecutionStatus,
    FileOperationType,
)


class TestBaseSchema:
    """Test cases for BaseSchema class."""
    
    def test_base_schema_configuration(self):
        """Test BaseSchema model configuration."""
        schema = BaseSchema()
        assert schema.model_config["from_attributes"] is True
    
    def test_datetime_serialization_with_value(self):
        """Test datetime serialization when value is present."""
        schema = BaseSchema()
        test_datetime = datetime(2024, 1, 15, 10, 30, 45)
        result = schema.serialize_datetime(test_datetime)
        assert result == "2024-01-15T10:30:45"
    
    def test_datetime_serialization_with_none(self):
        """Test datetime serialization when value is None."""
        schema = BaseSchema()
        result = schema.serialize_datetime(None)
        assert result is None
    
    def test_datetime_serialization_with_non_datetime(self):
        """Test datetime serialization with non-datetime value."""
        schema = BaseSchema()
        test_value = "not a datetime"
        result = schema.serialize_datetime(test_value)
        assert result == test_value


class TestUserProfileCreate:
    """Test cases for UserProfileCreate schema."""
    
    def test_valid_user_profile_create_minimal(self):
        """Test creating user profile with minimal required fields."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000"
        }
        profile = UserProfileCreate(**data)
        assert profile.user_id == data["user_id"]
        assert profile.email is None
        assert profile.name is None
        assert profile.avatar_url is None
        assert profile.preferences == {}
        assert profile.metadata == {}
    
    def test_valid_user_profile_create_complete(self):
        """Test creating user profile with all fields."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "test@example.com",
            "name": "Test User",
            "avatar_url": "https://example.com/avatar.jpg",
            "preferences": {"theme": "dark", "language": "en"},
            "metadata": {"source": "api", "version": "1.0"}
        }
        profile = UserProfileCreate(**data)
        assert profile.user_id == data["user_id"]
        assert profile.email == data["email"]
        assert profile.name == data["name"]
        assert profile.avatar_url == data["avatar_url"]
        assert profile.preferences == data["preferences"]
        assert profile.metadata == data["metadata"]
    
    def test_invalid_user_id_not_uuid(self):
        """Test validation error for non-UUID user_id."""
        data = {"user_id": "not-a-uuid"}
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(**data)
        assert "user_id must be a valid UUID" in str(exc_info.value)
    
    def test_invalid_user_id_empty(self):
        """Test validation error for empty user_id."""
        data = {"user_id": ""}
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(**data)
        assert "User ID cannot be empty" in str(exc_info.value)
    
    def test_invalid_email_format(self):
        """Test validation error for invalid email format."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "invalid-email"
        }
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(**data)
        assert "Invalid email format" in str(exc_info.value)
    
    def test_missing_required_user_id(self):
        """Test validation error when user_id is missing."""
        data = {"email": "test@example.com"}
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(**data)
        assert "Field required" in str(exc_info.value)


class TestUserProfileUpdate:
    """Test cases for UserProfileUpdate schema."""
    
    def test_valid_user_profile_update_partial(self):
        """Test updating user profile with partial fields."""
        data = {
            "email": "updated@example.com",
            "name": "Updated User"
        }
        profile = UserProfileUpdate(**data)
        assert profile.email == data["email"]
        assert profile.name == data["name"]
        assert profile.avatar_url is None
    
    def test_valid_user_profile_update_empty(self):
        """Test updating user profile with no fields (all optional)."""
        profile = UserProfileUpdate()
        assert profile.email is None
        assert profile.name is None
        assert profile.avatar_url is None
        assert profile.preferences is None
        assert profile.metadata is None
        assert profile.is_active is None
        assert profile.last_login is None
    
    def test_invalid_email_format_update(self):
        """Test validation error for invalid email in update."""
        data = {"email": "invalid-email"}
        with pytest.raises(ValidationError) as exc_info:
            UserProfileUpdate(**data)
        assert "Invalid email format" in str(exc_info.value)


class TestChatSessionCreate:
    """Test cases for ChatSessionCreate schema."""
    
    def test_valid_chat_session_create_minimal(self):
        """Test creating chat session with minimal required fields."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000"
        }
        session = ChatSessionCreate(**data)
        assert session.user_id == data["user_id"]
        assert session.title is None
        assert session.description is None
        assert session.context == {}
        assert session.settings == {}
        assert session.tags == []
    
    def test_valid_chat_session_create_complete(self):
        """Test creating chat session with all fields."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Session",
            "description": "A test chat session",
            "context": {"mode": "chat", "model": "gpt-4"},
            "settings": {"temperature": 0.7, "max_tokens": 1000},
            "tags": ["test", "development"]
        }
        session = ChatSessionCreate(**data)
        assert session.user_id == data["user_id"]
        assert session.title == data["title"]
        assert session.description == data["description"]
        assert session.context == data["context"]
        assert session.settings == data["settings"]
        assert session.tags == data["tags"]
    
    def test_invalid_user_id_not_uuid_session(self):
        """Test validation error for non-UUID user_id in session."""
        data = {"user_id": "not-a-uuid"}
        with pytest.raises(ValidationError) as exc_info:
            ChatSessionCreate(**data)
        assert "user_id must be a valid UUID" in str(exc_info.value)
    
    def test_invalid_title_whitespace_only(self):
        """Test validation error for title with only whitespace."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "   "  # Only whitespace
        }
        with pytest.raises(ValidationError) as exc_info:
            ChatSessionCreate(**data)
        assert "Title cannot be empty" in str(exc_info.value)
    
    def test_valid_title_empty_string(self):
        """Test that empty string title is allowed (becomes None after validation)."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": ""
        }
        session = ChatSessionCreate(**data)
        assert session.title == ""  # Empty string is preserved


class TestChatSessionUpdate:
    """Test cases for ChatSessionUpdate schema."""
    
    def test_valid_chat_session_update_partial(self):
        """Test updating chat session with partial fields."""
        data = {
            "title": "Updated Session",
            "is_active": False
        }
        session = ChatSessionUpdate(**data)
        assert session.title == data["title"]
        assert session.is_active == data["is_active"]
        assert session.description is None
    
    def test_invalid_title_update_whitespace(self):
        """Test validation error for title with only whitespace in update."""
        data = {"title": "   "}
        with pytest.raises(ValidationError) as exc_info:
            ChatSessionUpdate(**data)
        assert "Title cannot be empty" in str(exc_info.value)


class TestChatMessageCreate:
    """Test cases for ChatMessageCreate schema."""
    
    def test_valid_chat_message_create_minimal(self):
        """Test creating chat message with minimal required fields."""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "role": MessageRole.USER,
            "content": "Hello, world!",
            "sequence_number": 1
        }
        message = ChatMessageCreate(**data)
        assert message.session_id == data["session_id"]
        assert message.role == data["role"]
        assert message.content == data["content"]
        assert message.sequence_number == data["sequence_number"]
        assert message.metadata == {}
        assert message.tool_calls == []
        assert message.tool_results == []
    
    def test_valid_chat_message_create_complete(self):
        """Test creating chat message with all fields."""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "role": MessageRole.ASSISTANT,
            "content": "Hello! How can I help you?",
            "metadata": {"model": "gpt-4", "temperature": 0.7},
            "tool_calls": [{"name": "search", "args": {"query": "test"}}],
            "tool_results": [{"result": "success", "data": "found"}],
            "parent_message_id": "parent-123",
            "sequence_number": 2,
            "tokens_used": 150,
            "model_used": "gpt-4",
            "processing_time": 1.5
        }
        message = ChatMessageCreate(**data)
        assert message.session_id == data["session_id"]
        assert message.role == data["role"]
        assert message.content == data["content"]
        assert message.metadata == data["metadata"]
        assert message.tool_calls == data["tool_calls"]
        assert message.tool_results == data["tool_results"]
        assert message.parent_message_id == data["parent_message_id"]
        assert message.sequence_number == data["sequence_number"]
        assert message.tokens_used == data["tokens_used"]
        assert message.model_used == data["model_used"]
        assert message.processing_time == data["processing_time"]
    
    def test_invalid_session_id_not_uuid_message(self):
        """Test validation error for non-UUID session_id in message."""
        data = {
            "session_id": "not-a-uuid",
            "role": MessageRole.USER,
            "content": "Hello",
            "sequence_number": 1
        }
        with pytest.raises(ValidationError) as exc_info:
            ChatMessageCreate(**data)
        assert "session_id must be a valid UUID" in str(exc_info.value)
    
    def test_invalid_content_empty(self):
        """Test validation error for empty content."""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "role": MessageRole.USER,
            "content": "",
            "sequence_number": 1
        }
        with pytest.raises(ValidationError) as exc_info:
            ChatMessageCreate(**data)
        assert "Message content cannot be empty" in str(exc_info.value)
    
    def test_invalid_sequence_number_negative(self):
        """Test validation error for negative sequence number."""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "role": MessageRole.USER,
            "content": "Hello",
            "sequence_number": -1
        }
        with pytest.raises(ValidationError) as exc_info:
            ChatMessageCreate(**data)
        assert "Sequence number must be positive" in str(exc_info.value)
    
    def test_invalid_tokens_used_negative(self):
        """Test validation error for negative tokens used."""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "role": MessageRole.USER,
            "content": "Hello",
            "sequence_number": 1,
            "tokens_used": -10
        }
        with pytest.raises(ValidationError) as exc_info:
            ChatMessageCreate(**data)
        assert "Tokens used must be non-negative" in str(exc_info.value)
    
    def test_invalid_processing_time_negative(self):
        """Test validation error for negative processing time."""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "role": MessageRole.USER,
            "content": "Hello",
            "sequence_number": 1,
            "processing_time": -1.0
        }
        with pytest.raises(ValidationError) as exc_info:
            ChatMessageCreate(**data)
        assert "Processing time must be non-negative" in str(exc_info.value)


class TestChatMessageUpdate:
    """Test cases for ChatMessageUpdate schema."""
    
    def test_valid_chat_message_update_partial(self):
        """Test updating chat message with partial fields."""
        data = {
            "content": "Updated content",
            "tokens_used": 200
        }
        message = ChatMessageUpdate(**data)
        assert message.content == data["content"]
        assert message.tokens_used == data["tokens_used"]
        assert message.metadata is None
    
    def test_invalid_content_empty_update(self):
        """Test validation error for empty content in update."""
        data = {"content": ""}
        with pytest.raises(ValidationError) as exc_info:
            ChatMessageUpdate(**data)
        assert "Content cannot be empty" in str(exc_info.value)


class TestAgentTaskCreate:
    """Test cases for AgentTaskCreate schema."""
    
    def test_valid_agent_task_create_minimal(self):
        """Test creating agent task with minimal required fields."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something useful"
        }
        task = AgentTaskCreate(**data)
        assert task.user_id == data["user_id"]
        assert task.title == data["title"]
        assert task.instructions == data["instructions"]
        assert task.session_id is None
        assert task.description is None
        assert task.priority == 1
        assert task.context == {}
        assert task.requirements == []
        assert task.deliverables == []
        assert task.tags == []
        assert task.estimated_duration is None
    
    def test_valid_agent_task_create_complete(self):
        """Test creating agent task with all fields."""
        data = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "550e8400-e29b-41d4-a716-446655440001",
            "title": "Complex Task",
            "description": "A complex task description",
            "instructions": "Detailed instructions for the task",
            "priority": 5,
            "context": {"environment": "production", "version": "1.0"},
            "requirements": ["Python 3.8+", "Docker"],
            "deliverables": ["Code", "Documentation", "Tests"],
            "tags": ["urgent", "backend"],
            "estimated_duration": 120
        }
        task = AgentTaskCreate(**data)
        assert task.session_id == data["session_id"]
        assert task.user_id == data["user_id"]
        assert task.title == data["title"]
        assert task.description == data["description"]
        assert task.instructions == data["instructions"]
        assert task.priority == data["priority"]
        assert task.context == data["context"]
        assert task.requirements == data["requirements"]
        assert task.deliverables == data["deliverables"]
        assert task.tags == data["tags"]
        assert task.estimated_duration == data["estimated_duration"]
    
    def test_invalid_title_empty_task(self):
        """Test validation error for empty title in task."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "",
            "instructions": "Do something"
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskCreate(**data)
        assert "Task title cannot be empty" in str(exc_info.value)
    
    def test_invalid_instructions_empty(self):
        """Test validation error for empty instructions."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": ""
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskCreate(**data)
        assert "Task instructions cannot be empty" in str(exc_info.value)
    
    def test_invalid_priority_out_of_range_low(self):
        """Test validation error for priority below valid range."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something",
            "priority": 0
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskCreate(**data)
        assert "Priority must be between 1 and 5" in str(exc_info.value)
    
    def test_invalid_priority_out_of_range_high(self):
        """Test validation error for priority above valid range."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something",
            "priority": 6
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskCreate(**data)
        assert "Priority must be between 1 and 5" in str(exc_info.value)
    
    def test_invalid_estimated_duration_negative(self):
        """Test validation error for negative estimated duration."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something",
            "estimated_duration": -30
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskCreate(**data)
        assert "Estimated duration must be positive" in str(exc_info.value)


class TestAgentTaskUpdate:
    """Test cases for AgentTaskUpdate schema."""
    
    def test_valid_agent_task_update_partial(self):
        """Test updating agent task with partial fields."""
        data = {
            "status": TaskStatus.IN_PROGRESS,
            "progress": 50
        }
        task = AgentTaskUpdate(**data)
        assert task.status == data["status"]
        assert task.progress == data["progress"]
        assert task.title is None
    
    def test_invalid_progress_out_of_range_low(self):
        """Test validation error for progress below valid range."""
        data = {"progress": -1}
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskUpdate(**data)
        assert "Progress must be between 0 and 100" in str(exc_info.value)
    
    def test_invalid_progress_out_of_range_high(self):
        """Test validation error for progress above valid range."""
        data = {"progress": 101}
        with pytest.raises(ValidationError) as exc_info:
            AgentTaskUpdate(**data)
        assert "Progress must be between 0 and 100" in str(exc_info.value)


class TestAgentExecutionCreate:
    """Test cases for AgentExecutionCreate schema."""
    
    def test_valid_agent_execution_create_minimal(self):
        """Test creating agent execution with minimal required fields."""
        data = {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "step_name": "initialize",
            "step_type": "setup",
            "sequence_number": 1
        }
        execution = AgentExecutionCreate(**data)
        assert execution.task_id == data["task_id"]
        assert execution.step_name == data["step_name"]
        assert execution.step_type == data["step_type"]
        assert execution.sequence_number == data["sequence_number"]
        assert execution.input_data is None
    
    def test_invalid_step_name_empty(self):
        """Test validation error for empty step name."""
        data = {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "step_name": "",
            "step_type": "setup",
            "sequence_number": 1
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentExecutionCreate(**data)
        assert "Step name cannot be empty" in str(exc_info.value)
    
    def test_invalid_step_type_empty(self):
        """Test validation error for empty step type."""
        data = {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "step_name": "initialize",
            "step_type": "",
            "sequence_number": 1
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentExecutionCreate(**data)
        assert "Step type cannot be empty" in str(exc_info.value)
    
    def test_invalid_sequence_number_negative_execution(self):
        """Test validation error for negative sequence number in execution."""
        data = {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "step_name": "initialize",
            "step_type": "setup",
            "sequence_number": -1
        }
        with pytest.raises(ValidationError) as exc_info:
            AgentExecutionCreate(**data)
        assert "Sequence number must be positive" in str(exc_info.value)


class TestAgentExecutionUpdate:
    """Test cases for AgentExecutionUpdate schema."""
    
    def test_valid_agent_execution_update_partial(self):
        """Test updating agent execution with partial fields."""
        data = {
            "status": ExecutionStatus.COMPLETED,
            "duration": 45.5
        }
        execution = AgentExecutionUpdate(**data)
        assert execution.status == data["status"]
        assert execution.duration == data["duration"]
        assert execution.output_data is None
    
    def test_invalid_duration_negative_execution(self):
        """Test validation error for negative duration in execution."""
        data = {"duration": -10.0}
        with pytest.raises(ValidationError) as exc_info:
            AgentExecutionUpdate(**data)
        assert "Duration must be non-negative" in str(exc_info.value)
    
    def test_invalid_retry_count_negative(self):
        """Test validation error for negative retry count."""
        data = {"retry_count": -1}
        with pytest.raises(ValidationError) as exc_info:
            AgentExecutionUpdate(**data)
        assert "Retry count must be non-negative" in str(exc_info.value)


class TestFileOperationCreate:
    """Test cases for FileOperationCreate schema."""
    
    def test_valid_file_operation_create_minimal(self):
        """Test creating file operation with minimal required fields."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "operation_type": FileOperationType.CREATE,
            "file_path": "/path/to/file.txt"
        }
        operation = FileOperationCreate(**data)
        assert operation.user_id == data["user_id"]
        assert operation.operation_type == data["operation_type"]
        assert operation.file_path == data["file_path"]
        assert operation.session_id is None
        assert operation.task_id is None
        assert operation.old_path is None
        assert operation.file_size is None
        assert operation.file_hash is None
        assert operation.content_preview is None
        assert operation.metadata == {}
        assert operation.success is True
        assert operation.error_message is None
        assert operation.backup_path is None
        assert operation.changes_summary is None
    
    def test_invalid_file_path_empty(self):
        """Test validation error for empty file path."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "operation_type": FileOperationType.CREATE,
            "file_path": ""
        }
        with pytest.raises(ValidationError) as exc_info:
            FileOperationCreate(**data)
        assert "File path cannot be empty" in str(exc_info.value)
    
    def test_invalid_file_size_negative(self):
        """Test validation error for negative file size."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "operation_type": FileOperationType.CREATE,
            "file_path": "/path/to/file.txt",
            "file_size": -100
        }
        with pytest.raises(ValidationError) as exc_info:
            FileOperationCreate(**data)
        assert "File size must be non-negative" in str(exc_info.value)
    
    def test_content_preview_truncation(self):
        """Test content preview truncation for long content."""
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "operation_type": FileOperationType.CREATE,
            "file_path": "/path/to/file.txt",
            "content_preview": "x" * 1001  # Long content
        }
        # Should not raise error, but truncate content
        operation = FileOperationCreate(**data)
        assert len(operation.content_preview) == 1003  # 1000 + '...'
        assert operation.content_preview.endswith('...')


class TestFileOperationUpdate:
    """Test cases for FileOperationUpdate schema."""
    
    def test_valid_file_operation_update_partial(self):
        """Test updating file operation with partial fields."""
        data = {
            "success": False,
            "error_message": "File not found"
        }
        operation = FileOperationUpdate(**data)
        assert operation.success == data["success"]
        assert operation.error_message == data["error_message"]
        assert operation.file_size is None
    
    def test_invalid_file_size_negative_update(self):
        """Test validation error for negative file size in update."""
        data = {"file_size": -50}
        with pytest.raises(ValidationError) as exc_info:
            FileOperationUpdate(**data)
        assert "File size must be non-negative" in str(exc_info.value)


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions across all schemas."""
    
    def test_uuid_validation_with_various_formats(self):
        """Test UUID validation with different valid and invalid formats."""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "6ba7b811-9dad-11d1-80b4-00c04fd430c8"
        ]
        
        for uuid_str in valid_uuids:
            data = {"user_id": uuid_str}
            profile = UserProfileCreate(**data)
            assert profile.user_id == uuid_str
        
        # Test empty string validation for UserProfileCreate
        data = {"user_id": ""}
        with pytest.raises(ValidationError) as exc_info:
            UserProfileCreate(**data)
        assert "User ID cannot be empty" in str(exc_info.value)
        
        # Test UUID format validation for ChatSessionCreate
        invalid_uuids = [
            "not-a-uuid",
            "550e8400-e29b-41d4-a716",  # Too short
            "550e8400-e29b-41d4-a716-446655440000-extra",  # Too long
            "550e8400xe29bx41d4xa716x446655440000",  # Wrong format
        ]
        
        for uuid_str in invalid_uuids:
            data = {"user_id": uuid_str}
            with pytest.raises(ValidationError) as exc_info:
                ChatSessionCreate(**data)
            assert "user_id must be a valid UUID" in str(exc_info.value)
    
    def test_email_validation_edge_cases(self):
        """Test email validation with various edge cases."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org",
            "123@456.com"
        ]
        
        for email in valid_emails:
            data = {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": email
            }
            profile = UserProfileCreate(**data)
            assert profile.email == email
        
        # Test that None is allowed (optional field)
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "email": None
        }
        profile = UserProfileCreate(**data)
        assert profile.email is None
        
        # Test that empty string is allowed (validation only triggers if v is truthy)
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "email": ""
        }
        profile = UserProfileCreate(**data)
        assert profile.email == ""
        
        # Test invalid email formats (only when value is provided and truthy)
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "user@",
            "user@example",  # Missing TLD
            "user@.com"
        ]
        
        for email in invalid_emails:
            data = {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": email
            }
            with pytest.raises(ValidationError) as exc_info:
                UserProfileCreate(**data)
            assert "Invalid email format" in str(exc_info.value)
    
    def test_boundary_values_for_numeric_fields(self):
        """Test boundary values for numeric fields."""
        # Test priority boundaries
        for priority in [1, 2, 3, 4, 5]:  # Valid range
            data = {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Test Task",
                "instructions": "Do something",
                "priority": priority
            }
            task = AgentTaskCreate(**data)
            assert task.priority == priority
        
        # Test progress boundaries
        for progress in [0, 50, 100]:  # Valid range
            data = {"progress": progress}
            task_update = AgentTaskUpdate(**data)
            assert task_update.progress == progress
    
    def test_large_data_structures(self):
        """Test handling of large data structures in optional fields."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        large_list = [f"item_{i}" for i in range(1000)]
        
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something",
            "context": large_dict,
            "requirements": large_list,
            "deliverables": large_list,
            "tags": large_list
        }
        
        task = AgentTaskCreate(**data)
        assert len(task.context) == 1000
        assert len(task.requirements) == 1000
        assert len(task.deliverables) == 1000
        assert len(task.tags) == 1000
    
    def test_none_vs_empty_collections(self):
        """Test distinction between None and empty collections."""
        # Test with None (should use default_factory)
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something"
        }
        task = AgentTaskCreate(**data)
        assert task.context == {}  # default_factory dict
        assert task.requirements == []  # default_factory list
        
        # Test with explicit empty collections
        data_explicit = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Test Task",
            "instructions": "Do something",
            "context": {},
            "requirements": []
        }
        task_explicit = AgentTaskCreate(**data_explicit)
        assert task_explicit.context == {}
        assert task_explicit.requirements == []
    
    def test_special_characters_in_text_fields(self):
        """Test handling of special characters in text fields."""
        special_chars_text = "Test with émojis 🚀, symbols @#$%^&*(), and unicode ñáéíóú"
        
        data = {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": special_chars_text,
            "instructions": special_chars_text
        }
        
        task = AgentTaskCreate(**data)
        assert task.title == special_chars_text
        assert task.instructions == special_chars_text
    
    def test_datetime_serialization_edge_cases(self):
        """Test datetime serialization with edge cases."""
        schema = BaseSchema()
        
        # Test with microseconds
        dt_with_microseconds = datetime(2024, 1, 15, 10, 30, 45, 123456)
        result = schema.serialize_datetime(dt_with_microseconds)
        assert "2024-01-15T10:30:45.123456" == result
        
        # Test with timezone-naive datetime
        dt_naive = datetime(2024, 12, 31, 23, 59, 59)
        result = schema.serialize_datetime(dt_naive)
        assert "2024-12-31T23:59:59" == result
        
        # Test with various non-datetime types
        non_datetime_values = ["string", 123, [], {}, True, 0.5]
        for value in non_datetime_values:
            result = schema.serialize_datetime(value)
            assert result == value


class TestIntegrationScenarios:
    """Test integration scenarios that combine multiple schemas."""
    
    def test_complete_chat_workflow(self):
        """Test a complete chat workflow with session and messages."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        session_id = "550e8400-e29b-41d4-a716-446655440001"
        
        # Create user profile
        user_data = {
            "user_id": user_id,
            "email": "test@example.com",
            "name": "Test User"
        }
        user_profile = UserProfileCreate(**user_data)
        
        # Create chat session
        session_data = {
            "user_id": user_id,
            "title": "Test Chat Session",
            "description": "A test session for integration testing"
        }
        chat_session = ChatSessionCreate(**session_data)
        
        # Create messages
        message1_data = {
            "session_id": session_id,
            "role": MessageRole.USER,
            "content": "Hello, I need help with a task",
            "sequence_number": 1
        }
        message1 = ChatMessageCreate(**message1_data)
        
        message2_data = {
            "session_id": session_id,
            "role": MessageRole.ASSISTANT,
            "content": "I'd be happy to help! What do you need assistance with?",
            "sequence_number": 2,
            "parent_message_id": "msg-1"
        }
        message2 = ChatMessageCreate(**message2_data)
        
        # Verify all objects are created correctly
        assert user_profile.user_id == user_id
        assert chat_session.user_id == user_id
        assert message1.session_id == session_id
        assert message2.session_id == session_id
        assert message2.parent_message_id == "msg-1"
    
    def test_complete_task_workflow(self):
        """Test a complete task workflow with task and executions."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        task_id = "550e8400-e29b-41d4-a716-446655440002"
        
        # Create task
        task_data = {
            "user_id": user_id,
            "title": "Integration Test Task",
            "instructions": "Complete integration testing for schemas",
            "priority": 3,
            "estimated_duration": 60
        }
        task = AgentTaskCreate(**task_data)
        
        # Create execution steps
        execution1_data = {
            "task_id": task_id,
            "step_name": "initialize",
            "step_type": "setup",
            "sequence_number": 1
        }
        execution1 = AgentExecutionCreate(**execution1_data)
        
        execution2_data = {
            "task_id": task_id,
            "step_name": "execute",
            "step_type": "main",
            "sequence_number": 2
        }
        execution2 = AgentExecutionCreate(**execution2_data)
        
        # Update task progress
        task_update_data = {
            "status": TaskStatus.IN_PROGRESS,
            "progress": 50
        }
        task_update = AgentTaskUpdate(**task_update_data)
        
        # Update execution status
        execution_update_data = {
            "status": ExecutionStatus.COMPLETED,
            "duration": 30.5
        }
        execution_update = AgentExecutionUpdate(**execution_update_data)
        
        # Verify all objects are created correctly
        assert task.user_id == user_id
        assert task.priority == 3
        assert execution1.task_id == task_id
        assert execution2.task_id == task_id
        assert execution1.sequence_number == 1
        assert execution2.sequence_number == 2
        assert task_update.status == TaskStatus.IN_PROGRESS
        assert execution_update.status == ExecutionStatus.COMPLETED
    
    def test_file_operation_with_task_context(self):
        """Test file operations in the context of a task."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        task_id = "550e8400-e29b-41d4-a716-446655440003"
        session_id = "550e8400-e29b-41d4-a716-446655440004"
        
        # Create file operation
        file_op_data = {
            "session_id": session_id,
            "task_id": task_id,
            "user_id": user_id,
            "operation_type": FileOperationType.CREATE,
            "file_path": "/project/src/new_feature.py",
            "file_size": 1024,
            "content_preview": "def new_feature():\n    pass",
            "success": True,
            "changes_summary": "Created new feature module"
        }
        file_operation = FileOperationCreate(**file_op_data)
        
        # Update file operation
        file_update_data = {
            "file_size": 2048,
            "content_preview": "def new_feature():\n    return 'implemented'",
            "changes_summary": "Implemented new feature functionality"
        }
        file_update = FileOperationUpdate(**file_update_data)
        
        # Verify objects are created correctly
        assert file_operation.user_id == user_id
        assert file_operation.task_id == task_id
        assert file_operation.session_id == session_id
        assert file_operation.operation_type == FileOperationType.CREATE
        assert file_update.file_size == 2048
        assert "implemented" in file_update.content_preview