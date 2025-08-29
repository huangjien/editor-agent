"""Pydantic schemas for database operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, field_serializer

from .models import (
    ExecutionStatus,
    FileOperationType,
    MessageRole,
    TaskStatus,
)


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = {
        "from_attributes": True,
    }
    
    @field_serializer('*', when_used='json')
    def serialize_datetime(self, value):
        """Serialize datetime fields to ISO format"""
        if isinstance(value, datetime):
            return value.isoformat() if value else None
        return value


# User Profile Schemas
class UserProfileCreate(BaseSchema):
    """Schema for creating a user profile."""
    
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
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('User ID cannot be empty')
        # Validate UUID format
        try:
            UUID(v.strip())
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
        return v.strip()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, v):
                raise ValueError('Invalid email format')
        return v


class UserProfileUpdate(BaseSchema):
    """Schema for updating a user profile."""
    
    email: Optional[str] = Field(None, description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    metadata: Optional[Dict[str, Any]] = Field(None, description="User metadata")
    is_active: Optional[bool] = Field(None, description="User active status")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, v):
                raise ValueError('Invalid email format')
        return v


# Chat Session Schemas
class ChatSessionCreate(BaseSchema):
    """Schema for creating a chat session."""
    
    user_id: str = Field(..., description="User who owns this session")
    title: Optional[str] = Field(None, description="Session title")
    description: Optional[str] = Field(None, description="Session description")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Session context and state"
    )
    settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Session-specific settings"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Session tags"
    )
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        try:
            UUID(v)
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError('Title cannot be empty')
        return v.strip() if v else v


class ChatSessionUpdate(BaseSchema):
    """Schema for updating a chat session."""
    
    title: Optional[str] = Field(None, description="Session title")
    description: Optional[str] = Field(None, description="Session description")
    context: Optional[Dict[str, Any]] = Field(None, description="Session context")
    settings: Optional[Dict[str, Any]] = Field(None, description="Session settings")
    is_active: Optional[bool] = Field(None, description="Session active status")
    tags: Optional[List[str]] = Field(None, description="Session tags")
    last_activity: Optional[datetime] = Field(None, description="Last activity")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError('Title cannot be empty')
        return v.strip() if v else v


# Chat Message Schemas
class ChatMessageCreate(BaseSchema):
    """Schema for creating a chat message."""
    
    session_id: str = Field(..., description="Session this message belongs to")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Message metadata"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Tool calls made"
    )
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Tool call results"
    )
    parent_message_id: Optional[str] = Field(None, description="Parent message ID")
    sequence_number: int = Field(..., description="Message sequence")
    tokens_used: Optional[int] = Field(None, description="Tokens used")
    model_used: Optional[str] = Field(None, description="AI model used")
    processing_time: Optional[float] = Field(None, description="Processing time")
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v is not None:
            try:
                UUID(v)
            except ValueError:
                raise ValueError('session_id must be a valid UUID string')
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message content cannot be empty')
        return v
    
    @field_validator('sequence_number')
    @classmethod
    def validate_sequence_number(cls, v):
        if v <= 0:
            raise ValueError('Sequence number must be positive')
        return v
    
    @field_validator('tokens_used')
    @classmethod
    def validate_tokens_used(cls, v):
        if v is not None and v < 0:
            raise ValueError('Tokens used must be non-negative')
        return v
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v):
        if v is not None and v < 0:
            raise ValueError('Processing time must be non-negative')
        return v


class ChatMessageUpdate(BaseSchema):
    """Schema for updating a chat message."""
    
    content: Optional[str] = Field(None, description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Message metadata")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls")
    tool_results: Optional[List[Dict[str, Any]]] = Field(None, description="Tool results")
    tokens_used: Optional[int] = Field(None, description="Tokens used")
    model_used: Optional[str] = Field(None, description="AI model used")
    processing_time: Optional[float] = Field(None, description="Processing time")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v
    
    @field_validator('tokens_used')
    @classmethod
    def validate_tokens_used(cls, v):
        if v is not None and v < 0:
            raise ValueError('Tokens used must be non-negative')
        return v
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v):
        if v is not None and v < 0:
            raise ValueError('Processing time must be non-negative')
        return v


# Agent Task Schemas
class AgentTaskCreate(BaseSchema):
    """Schema for creating an agent task."""
    
    session_id: Optional[str] = Field(None, description="Associated chat session")
    user_id: str = Field(..., description="User who created the task")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    instructions: str = Field(..., description="Task instructions")
    priority: int = Field(1, description="Task priority (1-5)")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Task context"
    )
    requirements: Optional[List[str]] = Field(
        default_factory=list, description="Task requirements"
    )
    deliverables: Optional[List[str]] = Field(
        default_factory=list, description="Expected deliverables"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Task tags"
    )
    estimated_duration: Optional[int] = Field(
        None, description="Estimated duration in minutes"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Task title cannot be empty')
        return v.strip()
    
    @field_validator('instructions')
    @classmethod
    def validate_instructions(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Task instructions cannot be empty')
        return v.strip()
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Priority must be between 1 and 5')
        return v
    
    @field_validator('estimated_duration')
    @classmethod
    def validate_estimated_duration(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Estimated duration must be positive')
        return v


class AgentTaskUpdate(BaseSchema):
    """Schema for updating an agent task."""
    
    title: Optional[str] = Field(None, description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    instructions: Optional[str] = Field(None, description="Task instructions")
    status: Optional[TaskStatus] = Field(None, description="Task status")
    priority: Optional[int] = Field(None, description="Task priority")
    context: Optional[Dict[str, Any]] = Field(None, description="Task context")
    requirements: Optional[List[str]] = Field(None, description="Requirements")
    deliverables: Optional[List[str]] = Field(None, description="Deliverables")
    tags: Optional[List[str]] = Field(None, description="Task tags")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration")
    actual_duration: Optional[int] = Field(None, description="Actual duration")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    progress: Optional[int] = Field(None, description="Progress percentage")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError('Task title cannot be empty')
        return v.strip() if v else v
    
    @field_validator('instructions')
    @classmethod
    def validate_instructions(cls, v):
        if v and len(v.strip()) == 0:
            raise ValueError('Task instructions cannot be empty')
        return v.strip() if v else v
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError('Priority must be between 1 and 5')
        return v
    
    @field_validator('progress')
    @classmethod
    def validate_progress(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Progress must be between 0 and 100')
        return v
    
    @field_validator('estimated_duration', 'actual_duration')
    @classmethod
    def validate_duration(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Duration must be positive')
        return v


# Agent Execution Schemas
class AgentExecutionCreate(BaseSchema):
    """Schema for creating an agent execution record."""
    
    task_id: str = Field(..., description="Associated task ID")
    step_name: str = Field(..., description="Execution step name")
    step_type: str = Field(..., description="Type of execution step")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data")
    sequence_number: int = Field(..., description="Step sequence")
    
    @field_validator('step_name')
    @classmethod
    def validate_step_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Step name cannot be empty')
        return v.strip()
    
    @field_validator('step_type')
    @classmethod
    def validate_step_type(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Step type cannot be empty')
        return v.strip()
    
    @field_validator('sequence_number')
    @classmethod
    def validate_sequence_number(cls, v):
        if v <= 0:
            raise ValueError('Sequence number must be positive')
        return v


class AgentExecutionUpdate(BaseSchema):
    """Schema for updating an agent execution record."""
    
    status: Optional[ExecutionStatus] = Field(None, description="Execution status")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    retry_count: Optional[int] = Field(None, description="Retry count")
    logs: Optional[List[str]] = Field(None, description="Execution logs")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Execution metrics")
    
    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v):
        if v is not None and v < 0:
            raise ValueError('Duration must be non-negative')
        return v
    
    @field_validator('retry_count')
    @classmethod
    def validate_retry_count(cls, v):
        if v is not None and v < 0:
            raise ValueError('Retry count must be non-negative')
        return v


# File Operation Schemas
class FileOperationCreate(BaseSchema):
    """Schema for creating a file operation record."""
    
    session_id: Optional[str] = Field(None, description="Associated chat session")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    user_id: str = Field(..., description="User who performed the operation")
    operation_type: FileOperationType = Field(..., description="Operation type")
    file_path: str = Field(..., description="File path")
    old_path: Optional[str] = Field(None, description="Old path for rename/move")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File content hash")
    content_preview: Optional[str] = Field(None, description="Content preview")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Operation metadata"
    )
    success: bool = Field(True, description="Operation success")
    error_message: Optional[str] = Field(None, description="Error message")
    backup_path: Optional[str] = Field(None, description="Backup file path")
    changes_summary: Optional[str] = Field(None, description="Changes summary")
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('File path cannot be empty')
        return v.strip()
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        if v is not None and v < 0:
            raise ValueError('File size must be non-negative')
        return v
    
    @field_validator('content_preview')
    @classmethod
    def validate_content_preview(cls, v):
        if v and len(v) > 1000:
            return v[:1000] + '...'
        return v


class FileOperationUpdate(BaseSchema):
    """Schema for updating a file operation record."""
    
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="File content hash")
    content_preview: Optional[str] = Field(None, description="Content preview")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Operation metadata")
    success: Optional[bool] = Field(None, description="Operation success")
    error_message: Optional[str] = Field(None, description="Error message")
    backup_path: Optional[str] = Field(None, description="Backup file path")
    changes_summary: Optional[str] = Field(None, description="Changes summary")
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        if v is not None and v < 0:
            raise ValueError('File size must be non-negative')
        return v
    
    @field_validator('content_preview')
    @classmethod
    def validate_content_preview(cls, v):
        if v and len(v) > 1000:
            return v[:1000] + '...'
        return v