"""Tools and utilities for the LangGraph agent."""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseTool:
  """Base class for agent tools."""

  def __init__(self, name: str, description: str):
    self.name = name
    self.description = description

  async def execute(self, **kwargs) -> Dict[str, Any]:
    """Execute the tool with given parameters."""
    raise NotImplementedError

  def get_schema(self) -> Dict[str, Any]:
    """Get the tool's parameter schema."""
    return {"name": self.name, "description": self.description, "parameters": {}}


class FileReadTool(BaseTool):
  """Tool for reading file contents."""

  def __init__(self):
    super().__init__(name="read_file", description="Read the contents of a file")

  async def execute(self, file_path: str) -> Dict[str, Any]:
    """Read file contents."""
    try:
      path = Path(file_path)
      if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

      with open(path, "r", encoding="utf-8") as f:
        content = f.read()

      return {
        "success": True,
        "content": content,
        "file_path": str(path),
        "size": len(content),
      }
    except Exception as e:
      logger.error(f"Error reading file {file_path}: {str(e)}")
      return {"success": False, "error": str(e)}

  def get_schema(self) -> Dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["file_path"],
      },
    }


class FileWriteTool(BaseTool):
  """Tool for writing file contents."""

  def __init__(self):
    super().__init__(name="write_file", description="Write content to a file")

  async def execute(
    self, file_path: str, content: str, create_dirs: bool = True
  ) -> Dict[str, Any]:
    """Write content to file."""
    try:
      path = Path(file_path)

      # Create directories if needed
      if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

      with open(path, "w", encoding="utf-8") as f:
        f.write(content)

      return {
        "success": True,
        "file_path": str(path),
        "bytes_written": len(content.encode("utf-8")),
      }
    except Exception as e:
      logger.error(f"Error writing file {file_path}: {str(e)}")
      return {"success": False, "error": str(e)}

  def get_schema(self) -> Dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string", "description": "Path to the file to write"},
          "content": {"type": "string", "description": "Content to write to the file"},
          "create_dirs": {
            "type": "boolean",
            "description": "Whether to create parent directories",
            "default": True,
          },
        },
        "required": ["file_path", "content"],
      },
    }


class DirectoryListTool(BaseTool):
  """Tool for listing directory contents."""

  def __init__(self):
    super().__init__(
      name="list_directory", description="List the contents of a directory"
    )

  async def execute(
    self, directory_path: str, recursive: bool = False
  ) -> Dict[str, Any]:
    """List directory contents."""
    try:
      path = Path(directory_path)
      if not path.exists():
        return {"success": False, "error": f"Directory not found: {directory_path}"}

      if not path.is_dir():
        return {"success": False, "error": f"Path is not a directory: {directory_path}"}

      items = []
      if recursive:
        for item in path.rglob("*"):
          items.append(
            {
              "name": item.name,
              "path": str(item),
              "type": "directory" if item.is_dir() else "file",
              "size": item.stat().st_size if item.is_file() else None,
            }
          )
      else:
        for item in path.iterdir():
          items.append(
            {
              "name": item.name,
              "path": str(item),
              "type": "directory" if item.is_dir() else "file",
              "size": item.stat().st_size if item.is_file() else None,
            }
          )

      return {
        "success": True,
        "directory": str(path),
        "items": items,
        "count": len(items),
      }
    except Exception as e:
      logger.error(f"Error listing directory {directory_path}: {str(e)}")
      return {"success": False, "error": str(e)}

  def get_schema(self) -> Dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": {
        "type": "object",
        "properties": {
          "directory_path": {
            "type": "string",
            "description": "Path to the directory to list",
          },
          "recursive": {
            "type": "boolean",
            "description": "Whether to list recursively",
            "default": False,
          },
        },
        "required": ["directory_path"],
      },
    }


class CommandExecuteTool(BaseTool):
  """Tool for executing shell commands."""

  def __init__(self):
    super().__init__(name="execute_command", description="Execute a shell command")

  async def execute(
    self, command: str, working_dir: Optional[str] = None, timeout: int = 30
  ) -> Dict[str, Any]:
    """Execute shell command."""
    try:
      # Security check - basic command validation
      dangerous_commands = ["rm -rf", "sudo", "chmod 777", "mkfs", "dd if="]
      if any(dangerous in command.lower() for dangerous in dangerous_commands):
        return {
          "success": False,
          "error": "Command contains potentially dangerous operations",
        }

      cwd = Path(working_dir) if working_dir else None

      result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd
      )

      return {
        "success": result.returncode == 0,
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "working_dir": str(cwd) if cwd else None,
      }
    except subprocess.TimeoutExpired:
      return {"success": False, "error": f"Command timed out after {timeout} seconds"}
    except Exception as e:
      logger.error(f"Error executing command '{command}': {str(e)}")
      return {"success": False, "error": str(e)}

  def get_schema(self) -> Dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": {
        "type": "object",
        "properties": {
          "command": {"type": "string", "description": "Shell command to execute"},
          "working_dir": {
            "type": "string",
            "description": "Working directory for command execution",
          },
          "timeout": {
            "type": "integer",
            "description": "Timeout in seconds",
            "default": 30,
          },
        },
        "required": ["command"],
      },
    }


class SearchTool(BaseTool):
  """Tool for searching text in files."""

  def __init__(self):
    super().__init__(
      name="search_files", description="Search for text patterns in files"
    )

  async def execute(
    self,
    pattern: str,
    directory: str = ".",
    file_extensions: Optional[List[str]] = None,
  ) -> Dict[str, Any]:
    """Search for text patterns in files."""
    try:
      search_dir = Path(directory)
      if not search_dir.exists():
        return {"success": False, "error": f"Directory not found: {directory}"}

      matches = []
      file_patterns = [
        f"*.{ext}"
        for ext in (file_extensions or ["py", "txt", "md", "json", "yaml", "yml"])
      ]

      for file_pattern in file_patterns:
        for file_path in search_dir.rglob(file_pattern):
          if file_path.is_file():
            try:
              with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                  if pattern.lower() in line.lower():
                    matches.append(
                      {
                        "file": str(file_path),
                        "line_number": line_num,
                        "line_content": line.strip(),
                        "match_position": line.lower().find(pattern.lower()),
                      }
                    )
            except (UnicodeDecodeError, PermissionError):
              continue

      return {
        "success": True,
        "pattern": pattern,
        "directory": str(search_dir),
        "matches": matches,
        "total_matches": len(matches),
      }
    except Exception as e:
      logger.error(f"Error searching for pattern '{pattern}': {str(e)}")
      return {"success": False, "error": str(e)}

  def get_schema(self) -> Dict[str, Any]:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {"type": "string", "description": "Text pattern to search for"},
          "directory": {
            "type": "string",
            "description": "Directory to search in",
            "default": ".",
          },
          "file_extensions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "File extensions to search in",
          },
        },
        "required": ["pattern"],
      },
    }


# Tool registry
AVAILABLE_TOOLS = {
  "read_file": FileReadTool(),
  "write_file": FileWriteTool(),
  "list_directory": DirectoryListTool(),
  "execute_command": CommandExecuteTool(),
  "search_files": SearchTool(),
}


def get_available_tools(task_type: Optional[str] = None) -> List[str]:
  """Get list of available tools based on task type."""
  if task_type is None:
    return list(AVAILABLE_TOOLS.keys())

  # Filter tools based on task type
  task_lower = task_type.lower()

  if "read" in task_lower or "view" in task_lower or "show" in task_lower:
    return ["read_file", "list_directory", "search_files"]
  elif "write" in task_lower or "create" in task_lower or "edit" in task_lower:
    return ["write_file", "read_file", "list_directory"]
  elif "search" in task_lower or "find" in task_lower:
    return ["search_files", "list_directory"]
  elif "run" in task_lower or "execute" in task_lower:
    return ["execute_command", "read_file"]
  else:
    return list(AVAILABLE_TOOLS.keys())


async def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
  """Execute a tool by name with given parameters."""
  if tool_name not in AVAILABLE_TOOLS:
    return {
      "success": False,
      "error": f"Tool '{tool_name}' not found. Available tools: {list(AVAILABLE_TOOLS.keys())}",
    }

  tool = AVAILABLE_TOOLS[tool_name]
  try:
    result = await tool.execute(**kwargs)
    logger.info(f"Tool '{tool_name}' executed successfully")
    return result
  except Exception as e:
    logger.error(f"Error executing tool '{tool_name}': {str(e)}")
    return {"success": False, "error": str(e)}


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
  """Get the schema for a specific tool."""
  if tool_name in AVAILABLE_TOOLS:
    return AVAILABLE_TOOLS[tool_name].get_schema()
  return None


def get_all_tool_schemas() -> Dict[str, Dict[str, Any]]:
  """Get schemas for all available tools."""
  return {name: tool.get_schema() for name, tool in AVAILABLE_TOOLS.items()}
