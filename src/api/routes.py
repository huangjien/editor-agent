"""API routes for the editor agent."""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List

from src.api.schemas import (
  AgentRequest,
  AgentResponse,
  ChatMessage,
  ChatRequest,
  ChatResponse,
  HealthResponse,
)
from src.agent.service import AgentService
from src.utils.exceptions import ValidationError

api_router = APIRouter()
agent_router = APIRouter(prefix="/agent", tags=["agent"])
chat_router = APIRouter(prefix="/chat", tags=["chat"])


def get_agent_service() -> AgentService:
  """Dependency to get agent service instance."""
  return AgentService()


@agent_router.post("/execute", response_model=AgentResponse)
async def execute_agent(
  request: AgentRequest, agent_service: AgentService = Depends(get_agent_service)
) -> AgentResponse:
  """Execute an agent task."""
  try:
    result = await agent_service.execute_task(
      task=request.task, context=request.context, config=request.config
    )
    return AgentResponse(
      success=True, result=result, message="Task executed successfully"
    )
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@agent_router.get("/status")
async def get_agent_status(
  agent_service: AgentService = Depends(get_agent_service),
) -> Dict[str, Any]:
  """Get agent status and health information."""
  try:
    status = await agent_service.get_status()
    return {
      "status": "active",
      "agent_info": status,
      "timestamp": status.get("timestamp"),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@chat_router.post("/message", response_model=ChatResponse)
async def send_chat_message(
  request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)
) -> ChatResponse:
  """Send a chat message to the agent."""
  try:
    response = await agent_service.process_chat_message(
      message=request.message, session_id=request.session_id, context=request.context
    )
    return ChatResponse(
      message=response["message"],
      session_id=request.session_id,
      metadata=response.get("metadata", {}),
    )
  except ValidationError as e:
    raise HTTPException(status_code=422, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@chat_router.get("/sessions/{session_id}/history")
async def get_chat_history(
  session_id: str, agent_service: AgentService = Depends(get_agent_service)
) -> List[ChatMessage]:
  """Get chat history for a session."""
  try:
    history = await agent_service.get_chat_history(session_id)
    return [
      ChatMessage(
        content=msg["content"], role=msg["role"], timestamp=msg.get("timestamp")
      )
      for msg in history
    ]
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")


@chat_router.delete("/sessions/{session_id}")
async def clear_chat_session(
  session_id: str, agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, str]:
  """Clear a chat session."""
  try:
    await agent_service.clear_session(session_id)
    return {"message": f"Session {session_id} cleared successfully"}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


# Include sub-routers
api_router.include_router(agent_router)
api_router.include_router(chat_router)


@api_router.get("/health")
async def api_health_check() -> Dict[str, Any]:
  """API health check endpoint."""
  from src.utils.monitoring import get_health_checker
  
  health_checker = get_health_checker()
  return await health_checker.run_all_checks()


# Export router for main.py
router = api_router