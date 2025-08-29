"""Database models for Supabase integration."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FileOperationType(str, Enum):
    """File operation type enumeration."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"


class BaseDBModel(BaseModel):
    """Base model for database entities."""
    
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    model_config = {
        "from_attributes": True,
    }
    
    @field_serializer('created_at', 'updated_at', when_used='json')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime fields to ISO format for JSON output."""
        return value.isoformat() if value else None


class UserProfile(BaseDBModel):
    """User profile model."""
    
    user_id: str = Field(..., description="Unique user identifier")
    email: Optional[str] = Field(None, description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="User preferences and settings"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional user metadata"
    )
    is_active: bool = Field(True, description="Whether the user is active")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    @field_serializer('last_login', when_used='json')
    def serialize_last_login(self, value: datetime) -> str:
        """Serialize last_login field to ISO format for JSON output."""
        return value.isoformat() if value else None


class ChatSession(BaseDBModel):
    """Chat session model."""
    
    user_id: str = Field(..., description="User who owns this session")
    title: Optional[str] = Field(None, description="Session title")
    description: Optional[str] = Field(None, description="Session description")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Session context and state"
    )
    settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Session-specific settings"
    )
    is_active: bool = Field(True, description="Whether the session is active")
    message_count: int = Field(0, description="Number of messages in session")
    last_activity: Optional[datetime] = Field(
        None, description="Last activity timestamp"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Session tags for organization"
    )
    
    @field_serializer('last_activity', when_used='json')
    def serialize_last_activity(self, value: datetime) -> str:
        """Serialize last_activity field to ISO format for JSON output."""
        return value.isoformat() if value else None


class ChatMessage(BaseDBModel):
    """Chat message model."""
    
    session_id: str = Field(..., description="Session this message belongs to")
    role: MessageRole = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Message metadata (tokens, model, etc.)"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Tool calls made in this message"
    )
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Results from tool calls"
    )
    parent_message_id: Optional[str] = Field(
        None, description="Parent message ID for threading"
    )
    sequence_number: int = Field(..., description="Message sequence in session")
    tokens_used: Optional[int] = Field(None, description="Tokens used for this message")
    model_used: Optional[str] = Field(None, description="AI model used")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class AgentTask(BaseDBModel):
    """Agent task model."""
    
    session_id: Optional[str] = Field(None, description="Associated chat session")
    user_id: str = Field(..., description="User who created the task")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    instructions: str = Field(..., description="Task instructions for the agent")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current task status")
    priority: int = Field(1, description="Task priority (1-5, 5 being highest)")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Task context and parameters"
    )
    requirements: Optional[List[str]] = Field(
        default_factory=list, description="Task requirements"
    )
    deliverables: Optional[List[str]] = Field(
        default_factory=list, description="Expected deliverables"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Task tags for organization"
    )
    estimated_duration: Optional[int] = Field(
        None, description="Estimated duration in minutes"
    )
    actual_duration: Optional[int] = Field(
        None, description="Actual duration in minutes"
    )
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(
        None, description="Task result data"
    )
    progress: int = Field(0, description="Task progress percentage (0-100)")
    
    @field_serializer('started_at', 'completed_at', when_used='json')
    def serialize_task_datetime(self, value: datetime) -> str:
        """Serialize datetime fields to ISO format for JSON output."""
        return value.isoformat() if value else None


class AgentExecution(BaseDBModel):
    """Agent execution model for tracking individual execution steps."""
    
    task_id: str = Field(..., description="Associated task ID")
    step_name: str = Field(..., description="Execution step name")
    step_type: str = Field(..., description="Type of execution step")
    status: ExecutionStatus = Field(
        ExecutionStatus.STARTED, description="Execution status"
    )
    input_data: Optional[Dict[str, Any]] = Field(
        None, description="Input data for this step"
    )
    output_data: Optional[Dict[str, Any]] = Field(
        None, description="Output data from this step"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        None, description="Error details if step failed"
    )
    started_at: datetime = Field(..., description="Step start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Step completion timestamp")
    duration: Optional[float] = Field(None, description="Step duration in seconds")
    sequence_number: int = Field(..., description="Step sequence in task")
    retry_count: int = Field(0, description="Number of retries attempted")
    logs: Optional[List[str]] = Field(
        default_factory=list, description="Execution logs"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Execution metrics"
    )
    
    @field_serializer('started_at', 'completed_at', when_used='json')
    def serialize_execution_datetime(self, value: datetime) -> str:
        """Serialize datetime fields to ISO format for JSON output."""
        return value.isoformat() if value else None


class FileOperation(BaseDBModel):
    """File operation model for tracking file changes."""
    
    session_id: Optional[str] = Field(None, description="Associated chat session")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    user_id: str = Field(..., description="User who performed the operation")
    operation_type: FileOperationType = Field(
        ..., description="Type of file operation"
    )
    file_path: str = Field(..., description="File path that was operated on")
    old_path: Optional[str] = Field(
        None, description="Old path for rename/move operations"
    )
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File content hash")
    content_preview: Optional[str] = Field(
        None, description="Preview of file content (first 500 chars)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Operation metadata"
    )
    success: bool = Field(True, description="Whether operation was successful")
    error_message: Optional[str] = Field(
        None, description="Error message if operation failed"
    )
    backup_path: Optional[str] = Field(
        None, description="Backup file path if created"
    )
    changes_summary: Optional[str] = Field(
        None, description="Summary of changes made"
    )