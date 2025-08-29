"""Database module for Supabase integration."""

from .client import get_supabase_client, SupabaseClient
from .models import (
    ChatSession,
    ChatMessage,
    AgentTask,
    AgentExecution,
    UserProfile,
    FileOperation,
)
from .schemas import (
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatMessageCreate,
    ChatMessageUpdate,
    AgentTaskCreate,
    AgentTaskUpdate,
    AgentExecutionCreate,
    UserProfileCreate,
    UserProfileUpdate,
    FileOperationCreate,
)

__all__ = [
    "get_supabase_client",
    "SupabaseClient",
    "ChatSession",
    "ChatMessage",
    "AgentTask",
    "AgentExecution",
    "UserProfile",
    "FileOperation",
    "ChatSessionCreate",
    "ChatSessionUpdate",
    "ChatMessageCreate",
    "ChatMessageUpdate",
    "AgentTaskCreate",
    "AgentTaskUpdate",
    "AgentExecutionCreate",
    "UserProfileCreate",
    "UserProfileUpdate",
    "FileOperationCreate",
]