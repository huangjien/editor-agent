"""Supabase client implementation."""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

from ..config.settings import get_settings
from .schemas import (
    UserProfileCreate,
    ChatSessionCreate,
    ChatMessageCreate,
    AgentTaskCreate,
    AgentExecutionCreate,
)
from .models import (
    UserProfile,
    ChatSession,
    ChatMessage,
    AgentTask,
    AgentExecution,
)

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Wrapper for Supabase client with additional functionality."""

    def __init__(self, client: Client):
        self.client = client
        self._connected = False

    async def connect(self) -> bool:
        """Test connection to Supabase."""
        try:
            # Test connection by querying a system table
            self.client.rpc("version").execute()
            self._connected = True
            logger.info("Successfully connected to Supabase")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {str(e)}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._connected

    def test_connection(self) -> bool:
        """Test the Supabase connection."""
        try:
            # Simple query to test connection
            self.client.rpc("version").execute()
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Supabase connection."""
        from datetime import datetime
        try:
            # Simple query to test connection
            response = self.client.rpc("version").execute()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database": {
                    "connected": self._connected,
                    "version": response.data if response.data else "unknown",
                },
            }
        except Exception as e:
            logger.error(f"Supabase health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "database": {"connected": False},
            }

    # User Profiles
    def create_user_profile(self, profile_data: UserProfileCreate) -> UserProfile:
        """Create a new user profile."""
        try:
            response = self.client.table("user_profiles").insert(profile_data.model_dump()).execute()
            if response.data:
                return UserProfile(**response.data[0])
            raise ValueError("No data returned from user profile creation")
        except Exception as e:
            logger.error(f"Failed to create user profile: {str(e)}")
            raise

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by user ID."""
        try:
            response = (
                self.client.table("user_profiles")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            if response.data:
                return UserProfile(**response.data[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get user profile {user_id}: {str(e)}")
            return None

    def get_user_chat_sessions(self, user_id: str) -> List[ChatSession]:
        """Get chat sessions for a user."""
        try:
            response = (
                self.client.table("chat_sessions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            if response.data:
                return [ChatSession(**session) for session in response.data]
            return []
        except Exception as e:
            logger.error(f"Failed to get chat sessions for user {user_id}: {str(e)}")
            return []

    def get_user_agent_tasks(self, user_id: str) -> List[AgentTask]:
        """Get agent tasks for a user."""
        try:
            response = (
                self.client.table("agent_tasks")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            if response.data:
                return [AgentTask(**task) for task in response.data]
            return []
        except Exception as e:
            logger.error(f"Failed to get agent tasks for user {user_id}: {str(e)}")
            return []

    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """Get messages for a chat session."""
        try:
            response = (
                self.client.table("chat_messages")
                .select("*")
                .eq("session_id", session_id)
                .order("sequence_number")
                .execute()
            )
            if response.data:
                return [ChatMessage(**message) for message in response.data]
            return []
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {str(e)}")
            return []

    def get_task_executions(self, task_id: str) -> List[AgentExecution]:
        """Get executions for an agent task."""
        try:
            response = (
                self.client.table("agent_executions")
                .select("*")
                .eq("task_id", task_id)
                .order("created_at", desc=True)
                .execute()
            )
            if response.data:
                return [AgentExecution(**execution) for execution in response.data]
            return []
        except Exception as e:
            logger.error(f"Failed to get executions for task {task_id}: {str(e)}")
            return []

    # Chat Sessions
    def create_chat_session(self, session_data: ChatSessionCreate) -> ChatSession:
        """Create a new chat session."""
        try:
            response = self.client.table("chat_sessions").insert(session_data.model_dump()).execute()
            if response.data:
                return ChatSession(**response.data[0])
            raise ValueError("No data returned from chat session creation")
        except Exception as e:
            logger.error(f"Failed to create chat session: {str(e)}")
            raise

    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        try:
            response = (
                self.client.table("chat_sessions")
                .select("*")
                .eq("id", session_id)
                .execute()
            )
            if response.data:
                return ChatSession(**response.data[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {str(e)}")
            return None

    def update_chat_session(
        self, session_id: str, update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a chat session."""
        try:
            response = (
                self.client.table("chat_sessions")
                .update(update_data)
                .eq("id", session_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to update chat session {session_id}: {str(e)}")
            return None

    # Chat Messages
    def create_chat_message(self, message_data: ChatMessageCreate) -> ChatMessage:
        """Create a new chat message."""
        try:
            response = self.client.table("chat_messages").insert(message_data.model_dump()).execute()
            if response.data:
                return ChatMessage(**response.data[0])
            raise ValueError("No data returned from chat message creation")
        except Exception as e:
            logger.error(f"Failed to create chat message: {str(e)}")
            raise

    def get_chat_messages(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get chat messages for a session."""
        try:
            response = (
                self.client.table("chat_messages")
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=False)
                .limit(limit)
                .offset(offset)
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Failed to get chat messages for session {session_id}: {str(e)}")
            return []

    # Agent Tasks
    def create_agent_task(self, task_data: AgentTaskCreate) -> AgentTask:
        """Create a new agent task."""
        try:
            response = self.client.table("agent_tasks").insert(task_data.model_dump()).execute()
            if response.data:
                return AgentTask(**response.data[0])
            raise ValueError("No data returned from agent task creation")
        except Exception as e:
            logger.error(f"Failed to create agent task: {str(e)}")
            raise

    def get_agent_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent task by ID."""
        try:
            response = (
                self.client.table("agent_tasks")
                .select("*")
                .eq("id", task_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get agent task {task_id}: {str(e)}")
            return None

    def update_agent_task(
        self, task_id: str, update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update an agent task."""
        try:
            response = (
                self.client.table("agent_tasks")
                .update(update_data)
                .eq("id", task_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to update agent task {task_id}: {str(e)}")
            return None

    # Agent Executions
    def create_agent_execution(self, execution_data: AgentExecutionCreate) -> AgentExecution:
        """Create a new agent execution."""
        try:
            response = self.client.table("agent_executions").insert(execution_data.model_dump()).execute()
            if response.data:
                return AgentExecution(**response.data[0])
            raise ValueError("No data returned from agent execution creation")
        except Exception as e:
            logger.error(f"Failed to create agent execution: {str(e)}")
            raise

    def get_agent_executions(
        self, task_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get agent executions for a task."""
        try:
            response = (
                self.client.table("agent_executions")
                .select("*")
                .eq("task_id", task_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Failed to get agent executions for task {task_id}: {str(e)}")
            return []


@lru_cache()
def get_supabase_client() -> Optional[SupabaseClient]:
    """Get Supabase client instance."""
    settings = get_settings()
    
    if not settings.supabase_url or not settings.supabase_key:
        logger.warning("Supabase URL or key not configured")
        return None
    
    try:
        # Configure client options
        options = ClientOptions(
            auto_refresh_token=True,
            persist_session=True,
        )
        
        # Create Supabase client
        client = create_client(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_key,
            options=options,
        )
        
        return SupabaseClient(client)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {str(e)}")
        return None


def get_supabase_admin_client() -> Optional[SupabaseClient]:
    """Get Supabase admin client with service role key."""
    settings = get_settings()
    
    if not settings.supabase_url or not settings.supabase_service_key:
        logger.warning("Supabase URL or service key not configured")
        return None
    
    try:
        # Configure client options for admin
        options = ClientOptions(
            auto_refresh_token=False,
            persist_session=False,
        )
        
        # Create Supabase admin client
        client = create_client(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_service_key,
            options=options,
        )
        
        return SupabaseClient(client)
    except Exception as e:
        logger.error(f"Failed to create Supabase admin client: {str(e)}")
        return None