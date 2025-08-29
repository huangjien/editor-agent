"""Tests for agent tools."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agent.tools import (
  BaseTool,
  FileReadTool,
  FileWriteTool,
  DirectoryListTool,
  CommandExecuteTool,
  SearchTool,
  get_available_tools,
  get_all_tool_schemas,
  execute_tool,
  get_tool_schema,
)


class TestBaseTool:
  """Test BaseTool functionality."""

  def test_base_tool_creation(self):
    """Test creating a base tool."""
    tool = BaseTool(name="test_tool", description="A test tool")

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"

  @pytest.mark.asyncio
  async def test_base_tool_execute_not_implemented(self):
    """Test that base tool execute raises NotImplementedError."""
    tool = BaseTool("test", "test")

    with pytest.raises(NotImplementedError):
      await tool.execute()

  def test_base_tool_to_schema(self):
    """Test converting base tool to schema."""
    tool = BaseTool(name="test_tool", description="A test tool")

    schema = tool.get_schema()

    assert schema["name"] == "test_tool"
    assert schema["description"] == "A test tool"
    assert "parameters" in schema


class TestFileReadTool:
  """Test FileReadTool functionality."""

  def test_file_read_tool_creation(self):
    """Test creating a file read tool."""
    tool = FileReadTool()

    assert tool.name == "read_file"
    assert "Read the contents of a file" in tool.description
    assert "file_path" in tool.get_schema()["parameters"]["properties"]

  @pytest.mark.asyncio
  async def test_file_read_existing_file(self, sample_files):
    """Test reading an existing file."""
    tool = FileReadTool()

    result = await tool.execute(file_path=str(sample_files["text"]))

    assert result["success"] is True
    assert "This is a sample text file" in result["content"]
    assert "error" not in result

  @pytest.mark.asyncio
  async def test_file_read_nonexistent_file(self):
    """Test reading a nonexistent file."""
    tool = FileReadTool()

    result = await tool.execute(file_path="/nonexistent/file.txt")

    assert result["success"] is False
    assert "content" not in result
    assert "not found" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_file_read_directory(self, temp_dir):
    """Test reading a directory (should fail)."""
    tool = FileReadTool()

    result = await tool.execute(file_path=str(temp_dir))

    assert result["success"] is False
    assert "content" not in result
    assert "directory" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_file_read_binary_file(self, temp_dir):
    """Test reading a binary file."""
    # Create a binary file
    binary_file = temp_dir / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")

    tool = FileReadTool()
    result = await tool.execute(file_path=str(binary_file))

    # Should handle binary files gracefully
    assert result["success"] is True
    assert result["content"] is not None


class TestFileWriteTool:
  """Test FileWriteTool functionality."""

  def test_file_write_tool_creation(self):
    """Test creating a file write tool."""
    tool = FileWriteTool()

    assert tool.name == "write_file"
    assert "Write content to a file" in tool.description
    assert "file_path" in tool.get_schema()["parameters"]["properties"]
    assert "content" in tool.get_schema()["parameters"]["properties"]

  @pytest.mark.asyncio
  async def test_file_write_new_file(self, temp_dir):
    """Test writing to a new file."""
    tool = FileWriteTool()
    file_path = temp_dir / "new_file.txt"
    content = "Hello, World!"

    result = await tool.execute(file_path=str(file_path), content=content)

    assert result["success"] is True
    assert "error" not in result
    assert file_path.exists()
    assert file_path.read_text() == content

  @pytest.mark.asyncio
  async def test_file_write_existing_file(self, sample_files):
    """Test overwriting an existing file."""
    tool = FileWriteTool()
    new_content = "New content"

    result = await tool.execute(
      file_path=str(sample_files["text"]), content=new_content
    )

    assert result["success"] is True
    assert sample_files["text"].read_text() == new_content

  @pytest.mark.asyncio
  async def test_file_write_create_directories(self, temp_dir):
    """Test writing to a file in a new directory."""
    tool = FileWriteTool()
    file_path = temp_dir / "new_dir" / "nested" / "file.txt"
    content = "Nested file content"

    result = await tool.execute(file_path=str(file_path), content=content)

    assert result["success"] is True
    assert file_path.exists()
    assert file_path.read_text() == content

  @pytest.mark.asyncio
  async def test_file_write_permission_error(self):
    """Test writing to a file with permission error."""
    tool = FileWriteTool()

    # Try to write to a system directory (should fail)
    result = await tool.execute(file_path="/System/test.txt", content="test")

    assert result["success"] is False
    assert (
      "permission" in result["error"].lower()
      or "denied" in result["error"].lower()
      or "read-only" in result["error"].lower()
      or "not permitted" in result["error"].lower()
    )


class TestDirectoryListTool:
  """Test DirectoryListTool functionality."""

  def test_directory_list_tool_creation(self):
    """Test creating a directory list tool."""
    tool = DirectoryListTool()

    assert tool.name == "list_directory"
    assert "List the contents of a directory" in tool.description
    assert "directory_path" in tool.get_schema()["parameters"]["properties"]

  @pytest.mark.asyncio
  async def test_directory_list_existing_directory(self, temp_dir, sample_files):
    """Test listing an existing directory."""
    tool = DirectoryListTool()

    result = await tool.execute(directory_path=str(temp_dir))

    assert result["success"] is True
    assert isinstance(result["items"], list)
    assert len(result["items"]) > 0

    # Check that sample files are listed
    file_names = [item["name"] for item in result["items"]]
    assert "sample.txt" in file_names
    assert "sample.py" in file_names

  @pytest.mark.asyncio
  async def test_directory_list_nonexistent_directory(self):
    """Test listing a nonexistent directory."""
    tool = DirectoryListTool()

    result = await tool.execute(directory_path="/nonexistent/directory")

    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_directory_list_file_instead_of_directory(self, sample_files):
    """Test listing a file instead of a directory."""
    tool = DirectoryListTool()

    result = await tool.execute(directory_path=str(sample_files["text"]))

    assert result["success"] is False
    assert "not a directory" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_directory_list_with_details(self, temp_dir):
    """Test directory listing includes file details."""
    tool = DirectoryListTool()

    result = await tool.execute(directory_path=str(temp_dir))

    assert result["success"] is True

    # Check that items have required fields
    for item in result["items"]:
      assert "name" in item
      assert "type" in item
      assert "size" in item
      assert item["type"] in ["file", "directory"]


class TestCommandExecuteTool:
  """Test CommandExecuteTool functionality."""

  def test_command_execute_tool_creation(self):
    """Test creating a command execute tool."""
    tool = CommandExecuteTool()

    assert tool.name == "execute_command"
    assert "Execute a shell command" in tool.description
    schema = tool.get_schema()
    assert "command" in schema["parameters"]["properties"]

  @pytest.mark.asyncio
  async def test_command_execute_simple_command(self):
    """Test executing a simple command."""
    tool = CommandExecuteTool()

    result = await tool.execute(command="echo 'Hello, World!'")

    assert result["success"] is True
    assert "Hello, World!" in result["stdout"]
    assert result["return_code"] == 0

  @pytest.mark.asyncio
  async def test_command_execute_with_working_directory(self, temp_dir):
    """Test executing a command with working directory."""
    tool = CommandExecuteTool()

    result = await tool.execute(command="pwd", working_dir=str(temp_dir))

    assert result["success"] is True
    assert str(temp_dir) in result["stdout"]

  @pytest.mark.asyncio
  async def test_command_execute_failing_command(self):
    """Test executing a failing command."""
    tool = CommandExecuteTool()

    result = await tool.execute(command="nonexistent_command_12345")

    assert result["success"] is False
    assert result["return_code"] != 0
    assert result["stderr"] is not None

  @pytest.mark.asyncio
  async def test_command_execute_timeout(self):
    """Test command execution with timeout."""
    tool = CommandExecuteTool()

    result = await tool.execute(command="sleep 10", timeout=1)

    assert result["success"] is False
    assert "timed out" in result["error"].lower()


class TestSearchTool:
  """Test SearchTool functionality."""

  def test_search_tool_creation(self):
    """Test creating a search tool."""
    tool = SearchTool()

    assert tool.name == "search_files"
    assert "Search for text patterns in files" in tool.description
    schema = tool.get_schema()
    assert "pattern" in schema["parameters"]["properties"]

  @pytest.mark.asyncio
  async def test_search_by_filename(self, temp_dir, sample_files):
    """Test searching for content in files."""
    tool = SearchTool()

    result = await tool.execute(pattern="sample text file", directory=str(temp_dir))

    assert result["success"] is True
    assert len(result["matches"]) > 0
    assert any("sample.txt" in r["file"] for r in result["matches"])

  @pytest.mark.asyncio
  async def test_search_by_content(self, temp_dir, sample_files):
    """Test searching by file content."""
    tool = SearchTool()

    result = await tool.execute(pattern="sample text file", directory=str(temp_dir))

    assert result["success"] is True
    # Should find the sample.txt file
    assert len(result["matches"]) > 0

  @pytest.mark.asyncio
  async def test_search_by_extension(self, temp_dir, sample_files):
    """Test searching by file extension."""
    tool = SearchTool()

    result = await tool.execute(
      pattern="def", directory=str(temp_dir), file_extensions=["py"]
    )

    assert result["success"] is True
    assert len(result["matches"]) > 0
    assert all(r["file"].endswith(".py") for r in result["matches"])

  @pytest.mark.asyncio
  async def test_search_nonexistent_directory(self):
    """Test searching in a nonexistent directory."""
    tool = SearchTool()

    result = await tool.execute(pattern="test", directory="/nonexistent/directory")

    assert result["success"] is False
    assert "not found" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_search_with_max_results(self, temp_dir):
    """Test search with max results limit."""
    # Create multiple files
    for i in range(10):
      (temp_dir / f"test_file_{i}.txt").write_text(f"Content {i}")

    tool = SearchTool()

    result = await tool.execute(pattern="test_file", directory=str(temp_dir))

    assert result["success"] is True
    assert len(result["matches"]) <= 5


class TestToolRegistry:
  """Test tool registry functionality."""

  def test_get_available_tools(self):
    """Test getting available tools."""
    available_tools = get_available_tools()

    assert isinstance(available_tools, list)
    assert "read_file" in available_tools
    assert "write_file" in available_tools
    assert "list_directory" in available_tools
    assert "execute_command" in available_tools
    assert "search_files" in available_tools

  def test_get_all_tool_schemas(self):
    """Test getting all tool schemas."""
    schemas = get_all_tool_schemas()

    assert isinstance(schemas, dict)
    assert "read_file" in schemas
    assert "write_file" in schemas
    assert "list_directory" in schemas
    assert "execute_command" in schemas
    assert "search_files" in schemas
    assert len(schemas) > 0

    # Check schema structure
    for schema in schemas.values():
      assert "name" in schema
      assert "description" in schema
      assert "parameters" in schema

  @pytest.mark.asyncio
  async def test_execute_tool_existing(self, sample_files):
    """Test executing an existing tool."""
    result = await execute_tool("read_file", file_path=str(sample_files["text"]))

    assert result["success"] is True
    assert "sample text file" in result["content"]

  @pytest.mark.asyncio
  async def test_execute_tool_nonexistent(self):
    """Test executing a nonexistent tool."""
    result = await execute_tool("nonexistent_tool")

    assert result["success"] is False
    assert "not found" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_execute_tool_invalid_parameters(self):
    """Test executing a tool with invalid parameters."""
    # file_read requires 'path' parameter
    result = await execute_tool("read_file")

    assert result["success"] is False
    assert result["error"] is not None

  @pytest.mark.asyncio
  async def test_execute_tool_exception_handling(self):
    """Test tool execution exception handling."""
    # This should trigger an exception in the tool
    result = await execute_tool("read_file", file_path=None)

    assert result["success"] is False
    assert result["error"] is not None


class TestToolIntegration:
  """Test tool integration scenarios."""

  @pytest.mark.asyncio
  async def test_file_operations_workflow(self, temp_dir):
    """Test a complete file operations workflow."""
    file_path = temp_dir / "workflow_test.txt"
    content = "Test workflow content"

    # 1. Write a file
    write_result = await execute_tool(
      "write_file", file_path=str(file_path), content=content
    )
    assert write_result["success"] is True

    # 2. Read the file back
    read_result = await execute_tool("read_file", file_path=str(file_path))
    assert read_result["success"] is True
    assert read_result["content"] == content

    # 3. List the directory
    list_result = await execute_tool("list_directory", directory_path=str(temp_dir))
    assert list_result["success"] is True
    file_names = [item["name"] for item in list_result["items"]]
    assert "workflow_test.txt" in file_names

    # 4. Search for content in the file
    search_result = await execute_tool(
      "search_files", pattern="Test workflow", directory=str(temp_dir)
    )
    assert search_result["success"] is True
    assert len(search_result["matches"]) > 0

  @pytest.mark.asyncio
  async def test_command_and_file_integration(self, temp_dir):
    """Test integration between command execution and file operations."""
    # Create a file using command
    command_result = await execute_tool(
      "execute_command",
      command=f"echo 'Command created file' > {temp_dir}/command_file.txt",
      working_dir=str(temp_dir),
    )
    assert command_result["success"] is True

    # Read the file using file_read tool
    read_result = await execute_tool(
      "read_file", file_path=str(temp_dir / "command_file.txt")
    )
    assert read_result["success"] is True
    assert "Command created file" in read_result["content"]

  @pytest.mark.asyncio
  async def test_error_propagation(self):
    """Test that errors are properly propagated through the tool system."""
    # Try to read a nonexistent file
    result = await execute_tool("read_file", file_path="/definitely/does/not/exist.txt")

    assert result["success"] is False
    assert result["error"] is not None
    assert "content" not in result

    # Should propagate the error properly
    assert "error" in result
    assert "File not found" in result["error"] or "not found" in result["error"].lower()

    # Error should be descriptive
    assert len(result["error"]) > 0


class TestDirectoryListToolEdgeCases:
  """Test DirectoryListTool edge cases and error conditions."""

  @pytest.mark.asyncio
  async def test_directory_list_recursive_with_subdirectories(self, temp_dir):
    """Test recursive directory listing with nested subdirectories."""
    # Create nested directory structure
    (temp_dir / "level1").mkdir()
    (temp_dir / "level1" / "level2").mkdir()
    (temp_dir / "level1" / "level2" / "file.txt").write_text("nested file")
    (temp_dir / "level1" / "file1.txt").write_text("level1 file")
    (temp_dir / "root_file.txt").write_text("root file")

    tool = DirectoryListTool()
    result = await tool.execute(directory_path=str(temp_dir), recursive=True)

    assert result["success"] is True
    assert result["count"] >= 5  # At least 2 dirs + 3 files
    
    # Check that nested items are included
    file_paths = [item["path"] for item in result["items"]]
    assert any("level2" in path for path in file_paths)
    assert any("file.txt" in path for path in file_paths)

  @pytest.mark.asyncio
  async def test_directory_list_with_file_sizes(self, temp_dir):
    """Test that file sizes are correctly reported."""
    test_content = "This is test content for size calculation"
    test_file = temp_dir / "size_test.txt"
    test_file.write_text(test_content)
    
    tool = DirectoryListTool()
    result = await tool.execute(directory_path=str(temp_dir))
    
    assert result["success"] is True
    file_item = next((item for item in result["items"] if item["name"] == "size_test.txt"), None)
    assert file_item is not None
    assert file_item["size"] == len(test_content.encode('utf-8'))
    assert file_item["type"] == "file"


class TestCommandExecuteToolSecurity:
  """Test CommandExecuteTool security features and dangerous command detection."""

  @pytest.mark.asyncio
  async def test_dangerous_command_rm_rf(self):
    """Test that rm -rf commands are blocked."""
    tool = CommandExecuteTool()
    result = await tool.execute(command="rm -rf /some/path")
    
    assert result["success"] is False
    assert "dangerous operations" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_dangerous_command_sudo(self):
    """Test that sudo commands are blocked."""
    tool = CommandExecuteTool()
    result = await tool.execute(command="sudo rm file.txt")
    
    assert result["success"] is False
    assert "dangerous operations" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_dangerous_command_chmod_777(self):
    """Test that chmod 777 commands are blocked."""
    tool = CommandExecuteTool()
    result = await tool.execute(command="chmod 777 /etc/passwd")
    
    assert result["success"] is False
    assert "dangerous operations" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_dangerous_command_mkfs(self):
    """Test that mkfs commands are blocked."""
    tool = CommandExecuteTool()
    result = await tool.execute(command="mkfs.ext4 /dev/sda1")
    
    assert result["success"] is False
    assert "dangerous operations" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_dangerous_command_dd(self):
    """Test that dd if= commands are blocked."""
    tool = CommandExecuteTool()
    result = await tool.execute(command="dd if=/dev/zero of=/dev/sda")
    
    assert result["success"] is False
    assert "dangerous operations" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_safe_command_allowed(self):
    """Test that safe commands are allowed."""
    tool = CommandExecuteTool()
    result = await tool.execute(command="echo 'safe command'")
    
    assert result["success"] is True
    assert "safe command" in result["stdout"]


class TestSearchToolErrorHandling:
  """Test SearchTool error handling for various edge cases."""

  @pytest.mark.asyncio
  async def test_search_with_binary_files(self, temp_dir):
    """Test search behavior with binary files that cause UnicodeDecodeError."""
    # Create a binary file
    binary_file = temp_dir / "binary.bin"
    binary_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe\xfd')
    
    # Create a text file for comparison
    text_file = temp_dir / "text.txt"
    text_file.write_text("searchable text content")
    
    tool = SearchTool()
    result = await tool.execute(pattern="text", directory=str(temp_dir))
    
    # Should succeed and find the text file, skip the binary file
    assert result["success"] is True
    assert len(result["matches"]) >= 1
    assert any("text.txt" in match["file"] for match in result["matches"])

  @pytest.mark.asyncio
  async def test_search_empty_pattern(self, temp_dir, sample_files):
    """Test search with empty pattern."""
    tool = SearchTool()
    result = await tool.execute(pattern="", directory=str(temp_dir))
    
    # Should handle empty pattern gracefully
    assert result["success"] is True
    # Empty pattern should match every line
    assert len(result["matches"]) > 0

  @pytest.mark.asyncio
  async def test_search_case_sensitivity(self, temp_dir):
    """Test search case sensitivity."""
    test_file = temp_dir / "case_test.txt"
    test_file.write_text("This is a TEST file with Mixed Case")
    
    tool = SearchTool()
    
    # Search for lowercase should find uppercase
    result = await tool.execute(pattern="test", directory=str(temp_dir))
    assert result["success"] is True
    assert len(result["matches"]) > 0
    
    # Search for uppercase should find lowercase
    result = await tool.execute(pattern="MIXED", directory=str(temp_dir))
    assert result["success"] is True
    assert len(result["matches"]) > 0


class TestUtilityFunctions:
  """Test utility functions for tool management."""

  def test_get_available_tools_no_filter(self):
    """Test getting all available tools without filter."""
    tools = get_available_tools()
    expected_tools = ["read_file", "write_file", "list_directory", "execute_command", "search_files"]
    
    assert isinstance(tools, list)
    for tool in expected_tools:
      assert tool in tools

  def test_get_available_tools_read_task(self):
    """Test getting tools for read-related tasks."""
    tools = get_available_tools(task_type="read")
    assert "read_file" in tools
    assert "list_directory" in tools
    assert "search_files" in tools

  def test_get_available_tools_write_task(self):
    """Test getting tools for write-related tasks."""
    tools = get_available_tools(task_type="write")
    assert "write_file" in tools
    assert "read_file" in tools
    assert "list_directory" in tools

  def test_get_available_tools_search_task(self):
    """Test getting tools for search-related tasks."""
    tools = get_available_tools(task_type="search")
    assert "search_files" in tools
    assert "list_directory" in tools

  def test_get_available_tools_execute_task(self):
    """Test getting tools for execute-related tasks."""
    tools = get_available_tools(task_type="execute")
    assert "execute_command" in tools
    assert "read_file" in tools

  def test_get_available_tools_unknown_task(self):
    """Test getting tools for unknown task type."""
    tools = get_available_tools(task_type="unknown_task_type")
    # Should return all tools for unknown task types
    expected_tools = ["read_file", "write_file", "list_directory", "execute_command", "search_files"]
    for tool in expected_tools:
      assert tool in tools

  def test_get_tool_schema_valid_tool(self):
    """Test getting schema for a valid tool."""
    schema = get_tool_schema("read_file")
    assert schema is not None
    assert "name" in schema
    assert "description" in schema
    assert "parameters" in schema

  def test_get_tool_schema_invalid_tool(self):
    """Test getting schema for an invalid tool."""
    schema = get_tool_schema("nonexistent_tool")
    assert schema is None


class TestBoundaryConditions:
  """Test boundary conditions and edge cases."""

  @pytest.mark.asyncio
  async def test_file_read_very_long_path(self, temp_dir):
    """Test reading file with very long path."""
    # Create nested directory structure with long path
    long_path = temp_dir
    for i in range(10):
      long_path = long_path / f"very_long_directory_name_{i}"
      long_path.mkdir(exist_ok=True)
    
    test_file = long_path / "test_file.txt"
    test_content = "Content in deeply nested file"
    test_file.write_text(test_content)
    
    tool = FileReadTool()
    result = await tool.execute(file_path=str(test_file))
    
    assert result["success"] is True
    assert result["content"] == test_content

  @pytest.mark.asyncio
  async def test_file_write_empty_content(self, temp_dir):
    """Test writing empty content to file."""
    test_file = temp_dir / "empty_file.txt"
    
    tool = FileWriteTool()
    result = await tool.execute(file_path=str(test_file), content="")
    
    assert result["success"] is True
    assert test_file.exists()
    assert test_file.read_text() == ""

  @pytest.mark.asyncio
  async def test_search_very_long_pattern(self, temp_dir):
    """Test search with very long pattern."""
    test_file = temp_dir / "pattern_test.txt"
    long_pattern = "a" * 1000
    test_content = f"Start {long_pattern} End"
    test_file.write_text(test_content)
    
    tool = SearchTool()
    result = await tool.execute(pattern=long_pattern, directory=str(temp_dir))
    
    assert result["success"] is True
    assert len(result["matches"]) == 1
    assert long_pattern in result["matches"][0]["line_content"]

  @pytest.mark.asyncio
  async def test_command_execute_empty_command(self):
    """Test executing empty command."""
    tool = CommandExecuteTool()
    result = await tool.execute(command="")
    
    assert result["success"] is True
    assert result["return_code"] == 0
    assert result["stdout"] == ""
    assert result["stderr"] == ""

  @pytest.mark.asyncio
  async def test_directory_list_empty_directory(self, temp_dir):
    """Test listing empty directory."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()
    
    tool = DirectoryListTool()
    result = await tool.execute(directory_path=str(empty_dir))
    
    assert result["success"] is True
    assert result["count"] == 0
    assert len(result["items"]) == 0


class TestPerformanceScenarios:
  """Test performance-related scenarios."""

  def setup_method(self):
    self.temp_dir = tempfile.mkdtemp()
    self.test_dir = Path(self.temp_dir)
    self.file_read_tool = FileReadTool()
    self.file_write_tool = FileWriteTool()
    self.directory_list_tool = DirectoryListTool()
    self.search_tool = SearchTool()

  def teardown_method(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  @pytest.mark.asyncio
  async def test_directory_list_many_files(self, temp_dir):
    """Test listing directory with many files."""
    # Create 100 files
    for i in range(100):
      (temp_dir / f"file_{i:03d}.txt").write_text(f"Content {i}")
    
    tool = DirectoryListTool()
    result = await tool.execute(directory_path=str(temp_dir))
    
    assert result["success"] is True
    assert result["count"] == 100
    assert len(result["items"]) == 100

  @pytest.mark.asyncio
  async def test_search_many_files(self, temp_dir):
    """Test searching across many files."""
    # Create 50 files with searchable content
    for i in range(50):
      content = f"File {i} contains searchable content"
      (temp_dir / f"search_file_{i}.txt").write_text(content)
    
    tool = SearchTool()
    result = await tool.execute(pattern="searchable", directory=str(temp_dir))
    
    assert result["success"] is True
    assert len(result["matches"]) == 50

  @pytest.mark.asyncio
  async def test_search_large_file(self, temp_dir):
    """Test searching in a large file."""
    # Create a large file with repeated content
    large_content = "\n".join([f"Line {i} with searchable text" for i in range(1000)])
    large_file = temp_dir / "large_file.txt"
    large_file.write_text(large_content)
    
    tool = SearchTool()
    result = await tool.execute(pattern="searchable", directory=str(temp_dir))
    
    assert result["success"] is True
    assert len(result["matches"]) == 1000  # Should find all 1000 lines

  @pytest.mark.asyncio
  async def test_file_write_large_content(self, temp_dir):
    """Test writing large content to file."""
    large_content = "Large content line\n" * 10000
    test_file = temp_dir / "large_content.txt"
    
    tool = FileWriteTool()
    result = await tool.execute(file_path=str(test_file), content=large_content)
    
    assert result["success"] is True
    assert test_file.exists()
    
    # Verify content was written correctly
    tool_read = FileReadTool()
    read_result = await tool_read.execute(file_path=str(test_file))
    assert read_result["success"] is True
    assert read_result["content"] == large_content


class TestExceptionCoverage:
  """Test exception handling paths for complete coverage."""

  def setup_method(self):
    self.temp_dir = tempfile.mkdtemp()
    self.test_dir = Path(self.temp_dir)
    self.directory_list_tool = DirectoryListTool()
    self.command_tool = CommandExecuteTool()
    self.search_tool = SearchTool()

  def teardown_method(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  @pytest.mark.asyncio
  async def test_directory_list_general_exception(self):
    """Test DirectoryListTool general exception handling."""
    with patch('pathlib.Path.exists', side_effect=Exception("Unexpected error")):
      result = await self.directory_list_tool.execute(
        directory_path=str(self.test_dir)
      )
      assert result["success"] is False
      assert "error" in result
      assert "Unexpected error" in result["error"]

  @pytest.mark.asyncio
  async def test_command_execute_general_exception(self):
    """Test CommandExecuteTool general exception handling."""
    with patch('subprocess.run', side_effect=Exception("System error")):
      result = await self.command_tool.execute(command="echo test")
      assert result["success"] is False
      assert "error" in result
      assert "System error" in result["error"]

  @pytest.mark.asyncio
  async def test_search_tool_general_exception(self):
    """Test SearchTool general exception handling."""
    with patch('pathlib.Path.exists', side_effect=Exception("Path error")):
      result = await self.search_tool.execute(
        pattern="test", directory=str(self.test_dir)
      )
      assert result["success"] is False
      assert "error" in result
      assert "Path error" in result["error"]

  @pytest.mark.asyncio
  async def test_search_tool_unicode_decode_error(self):
    """Test SearchTool handling of UnicodeDecodeError."""
    # Create a binary file that will cause UnicodeDecodeError
    binary_file = self.test_dir / "binary.py"
    binary_file.write_bytes(b'\x80\x81\x82\x83')  # Invalid UTF-8
    
    result = await self.search_tool.execute(
      pattern="test", directory=str(self.test_dir)
    )
    # Should succeed but skip the binary file
    assert result["success"] is True
    assert result["total_matches"] == 0

  @pytest.mark.asyncio
  async def test_search_tool_permission_error(self):
    """Test SearchTool handling of PermissionError."""
    # Create a file and simulate permission error
    test_file = self.test_dir / "restricted.py"
    test_file.write_text("test content")
    
    with patch('builtins.open', side_effect=PermissionError("Access denied")):
      result = await self.search_tool.execute(
        pattern="test", directory=str(self.test_dir)
      )
      # Should succeed but skip files with permission errors
      assert result["success"] is True
      assert result["total_matches"] == 0
