"""Pydantic schemas for API request and response models."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


def utc_now() -> datetime:
  """Get current UTC datetime in a timezone-aware format."""
  return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
  """Chat message model."""

  content: str = Field(..., description="Message content")
  role: str = Field(..., description="Message role (user, assistant, system)")
  timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")


class ChatRequest(BaseModel):
  """Chat request model."""

  message: str = Field(..., description="User message")
  session_id: str = Field(..., description="Chat session ID")
  context: Optional[Dict[str, Any]] = Field(
    default=None, description="Additional context"
  )


class ChatResponse(BaseModel):
  """Chat response model."""

  message: str = Field(..., description="Agent response message")
  session_id: str = Field(..., description="Chat session ID")
  metadata: Optional[Dict[str, Any]] = Field(
    default=None, description="Response metadata"
  )


class AgentRequest(BaseModel):
  """Agent task execution request."""

  task: str = Field(..., description="Task description or instruction")
  context: Optional[Dict[str, Any]] = Field(default=None, description="Task context")
  config: Optional[Dict[str, Any]] = Field(
    default=None, description="Agent configuration"
  )


class AgentResponse(BaseModel):
  """Agent task execution response."""

  success: bool = Field(..., description="Whether the task was successful")
  result: Optional[Dict[str, Any]] = Field(default=None, description="Task result")
  message: str = Field(..., description="Response message")
  execution_time: Optional[float] = Field(
    default=None, description="Execution time in seconds"
  )


class HealthResponse(BaseModel):
  """Health check response."""

  status: str = Field(..., description="Service status")
  version: str = Field(..., description="Service version")
  service: str = Field(..., description="Service name")
  timestamp: Optional[datetime] = Field(
    default_factory=utc_now, description="Check timestamp"
  )


class ErrorResponse(BaseModel):
  """Error response model."""

  error: str = Field(..., description="Error message")
  detail: Optional[str] = Field(default=None, description="Error details")
  code: Optional[str] = Field(default=None, description="Error code")
  timestamp: datetime = Field(
    default_factory=utc_now, description="Error timestamp"
  )


class AgentConfig(BaseModel):
  """Agent configuration model."""

  model_name: Optional[str] = Field(default="gpt-4", description="LLM model name")
  temperature: Optional[float] = Field(
    default=0.7, ge=0.0, le=2.0, description="Model temperature"
  )
  max_tokens: Optional[int] = Field(default=1000, gt=0, description="Maximum tokens")
  timeout: Optional[int] = Field(
    default=30, gt=0, description="Request timeout in seconds"
  )
  streaming: Optional[bool] = Field(
    default=False, description="Enable streaming responses"
  )


class SessionInfo(BaseModel):
  """Chat session information."""

  session_id: str = Field(..., description="Session ID")
  created_at: datetime = Field(..., description="Session creation time")
  last_activity: datetime = Field(..., description="Last activity time")
  message_count: int = Field(..., description="Number of messages in session")
  metadata: Optional[Dict[str, Any]] = Field(
    default=None, description="Session metadata"
  )


class AgentStatus(BaseModel):
  """Agent status information."""

  status: str = Field(..., description="Agent status")
  active_sessions: int = Field(..., description="Number of active sessions")
  total_requests: int = Field(..., description="Total requests processed")
  uptime: float = Field(..., description="Uptime in seconds")
  last_activity: Optional[datetime] = Field(
    default=None, description="Last activity time"
  )
  memory_usage: Optional[Dict[str, Any]] = Field(
    default=None, description="Memory usage stats"
  )