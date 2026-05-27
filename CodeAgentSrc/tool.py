"""
tool.py - 工具系统模块
======================

本模块定义了 Agent 可用的所有工具及其实现，包括：
- 文件操作：read_file, write_file, edit_file
- 文件检索：list_files, grep_search
- Shell 命令：run_shell
- 网页获取：web_fetch
- 子 Agent 调用：agent（实际在 agent.py 中实现）

每个工具都有完整的错误处理和权限检查机制。
"""

import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import html as html_module
from pathlib import Path
from .memory import get_memory_dir, update_memory_index


# HTML 清理相关的正则表达式
_RE_SCRIPT = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL)
_RE_STYLE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")

# 文件搜索时的忽略配置
_IGNORE_DIRS = {".git", ".svn", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "node_modules", ".idea", ".vscode", "build", "dist", ".eggs"}
_IGNORE_EXTS = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".jpg", ".png", ".gif", ".ico", ".zip", ".tar", ".gz", ".DS_Store", "Thumbs.db"}
_FILE_ENCODINGS = ["utf-8", "gbk", "gb2312", "latin-1"]


# OpenAI Function Calling 格式的工具定义
tool_definitions: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容，返回带行号的文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "要读取的文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件内容，文件不存在则创建，已存在则覆盖",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "要写入的文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "编辑文件，通过精确匹配字符串替换内容，old_string 必须完全匹配（包括空格和缩进）",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "要编辑的文件路径"},
                    "old_string": {"type": "string", "description": "要查找并替换的精确字符串"},
                    "new_string": {"type": "string", "description": "用来替换的新字符串"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出匹配 glob 模式的文件，返回匹配的文件路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": '匹配文件的 glob 模式（例如 "**/*.ts"、"src/**/*"）'},
                    "path": {"type": "string", "description": "搜索的基础目录，默认为当前目录"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "在文件中搜索模式，返回匹配的行及文件路径和行号",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "要搜索的正则表达式模式"},
                    "path": {"type": "string", "description": "要搜索的目录或文件，默认为当前目录"},
                    "include": {"type": "string", "description": '要包含的文件 glob 模式（例如 "*.ts"、"*.py"）'},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "执行 shell 命令并返回输出，用于运行测试、安装包、git 操作等",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 shell 命令"},
                    "timeout": {"type": "number", "description": "超时时间（毫秒，默认 30000）"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "获取 URL 内容并以文本返回，对于 HTML 页面会去除标签返回可读文本，JSON/文本响应直接返回",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要获取的 URL"},
                    "max_length": {"type": "number", "description": "内容最大长度（字符，默认 50000）"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent",
            "description": "启动子 agent 自主处理任务，子 agent 有独立上下文并返回结果，类型：'explore'（只读）、'plan'（只读、结构化规划）、'general'（完整工具）",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "子 agent 任务的简短描述（3-5 字）"},
                    "prompt": {"type": "string", "description": "子 agent 的详细任务指令"},
                    "type": {"type": "string", "enum": ["explore", "plan", "general"], "description": "agent 类型，默认 general"},
                },
                "required": ["description", "prompt"],
            },
        },
    },
]


def _read_file(arguments: dict) -> str:
    """
    读取文件内容，返回带行号的格式
    
    参数:
        arguments: 包含 file_path 的字典
        
    返回:
        带行号的文件内容，或错误信息
    """
    file_path = arguments.get("file_path")
    if not file_path:
        return "错误：缺少必需参数 file_path"
    try:
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                lines.append(f"{i:4d}| {line}")
        content = "".join(lines)
        return content if content else "(空文件)"
    except FileNotFoundError:
        return f"错误：文件不存在: {file_path}"
    except PermissionError:
        return f"错误：没有读取权限: {file_path}"
    except OSError as e:
        return f"错误：读取文件失败: {e}"


def _write_file(arguments: dict) -> str:
    """
    写入文件内容，文件不存在则创建，已存在则覆盖
    
    如果写入到记忆目录，会自动更新 MEMORY.md 索引。
    
    参数:
        arguments: 包含 file_path 和 content 的字典
        
    返回:
        成功信息或错误信息
    """
    file_path = arguments.get("file_path")
    content = arguments.get("content")
    if not file_path:
        return "错误：缺少必需参数 file_path"
    if content is None:
        return "错误：缺少必需参数 content"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # 检测是否写入记忆目录
        memory_dir = get_memory_dir()
        file_path_obj = Path(file_path)
        try:
            file_path_obj.relative_to(memory_dir)
            # 是记忆文件，更新索引
            update_memory_index()
        except ValueError:
            # 不是记忆目录下的文件
            pass
        
        return f"成功写入文件: {file_path}"
    except PermissionError:
        return f"错误：没有写入权限: {file_path}"
    except OSError as e:
        return f"错误：写入文件失败: {e}"


def _edit_file(arguments: dict) -> str:
    """
    编辑文件，通过精确匹配字符串替换内容
    
    old_string 必须完全匹配（包括空格和缩进），且只能有一处匹配。
    如果编辑的是记忆文件，会自动更新 MEMORY.md 索引。
    
    参数:
        arguments: 包含 file_path, old_string, new_string 的字典
        
    返回:
        成功信息或错误信息
    """
    file_path = arguments.get("file_path")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    if not file_path:
        return "错误：缺少必需参数 file_path"
    if old_string is None:
        return "错误：缺少必需参数 old_string"
    if new_string is None:
        return "错误：缺少必需参数 new_string"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        count = content.count(old_string)
        if count == 0:
            return f"错误：未找到匹配的字符串 \"{old_string}\"，请检查参数是否正确"
        if count > 1:
            return f"错误：找到 {count} 处匹配 \"{old_string}\"，请提供更唯一的匹配点（需包含更多上下文）"
        new_content = content.replace(old_string, new_string, 1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        # 检测是否编辑记忆文件，是则更新索引
        memory_dir = get_memory_dir()
        file_path_obj = Path(file_path)
        try:
            file_path_obj.relative_to(memory_dir)
            # 是记忆文件，更新索引
            update_memory_index()
        except ValueError:
            # 不是记忆目录下的文件
            pass
        
        return f"成功编辑文件: {file_path}"
    except PermissionError:
        return f"错误：没有写入权限: {file_path}"
    except OSError as e:
        return f"错误：编辑文件失败: {e}"


def _list_files(arguments: dict) -> str:
    """
    列出匹配 glob 模式的文件
    
    参数:
        arguments: 包含 pattern 和可选 path 的字典
        
    返回:
        匹配的文件路径列表，或错误信息
    """
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    if not pattern:
        return "错误：缺少必需参数 pattern"
    try:
        matches = list(Path(path).glob(pattern))
        if not matches:
            return "(无匹配文件)"
        return "\n".join(str(m) for m in matches)
    except PermissionError:
        return f"错误：没有访问权限: {path}"
    except OSError as e:
        return f"错误：搜索文件失败: {e}"


def _grep_search(arguments: dict) -> str:
    """
    在文件中搜索正则表达式模式
    
    会自动忽略常见的二进制文件和目录，支持多种编码。
    
    参数:
        arguments: 包含 pattern 和可选 path, include 的字典
        
    返回:
        匹配的行（格式：文件名:行号: 内容），或错误信息
    """
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    include = arguments.get("include")
    if not pattern:
        return "错误：缺少必需参数 pattern"
    try:
        regex = re.compile(pattern)
        matches = []
        for file_path in Path(path).rglob(include if include and include.strip() else "*"):
            if not file_path.is_file():
                continue
            if any(part in _IGNORE_DIRS for part in file_path.parts):
                continue
            if file_path.suffix.lower() in _IGNORE_EXTS:
                continue
            try:
                with open(file_path, "rb") as f:
                    header = f.read(1024)
                if b"\x00" in header:
                    continue
            except PermissionError:
                continue
            content = None
            for enc in _FILE_ENCODINGS:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, LookupError, PermissionError):
                    continue
            if content is None:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{file_path}:{i}: {line.rstrip()}")
        return "\n".join(matches) if matches else "(无匹配)"
    except re.error as e:
        return f"错误：正则表达式无效: {e}"
    except OSError as e:
        return f"错误：搜索失败: {e}"


def _run_shell(arguments: dict) -> str:
    """
    执行 shell 命令并返回输出
    
    如果命令可能删除了记忆文件，会自动更新 MEMORY.md 索引。
    
    参数:
        arguments: 包含 command 和可选 timeout 的字典
        
    返回:
        命令输出（stdout + stderr），或错误信息
    """
    command = arguments.get("command")
    timeout_ms = arguments.get("timeout", 30000)
    if not command:
        return "错误：缺少必需参数 command"
    timeout_sec = timeout_ms / 1000
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=timeout_sec
        )
        parts = []
        if result.stdout:
            try:
                stdout_text = result.stdout.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                stdout_text = result.stdout.decode("gbk", errors="replace")
            parts.append(stdout_text)
        if result.stderr:
            try:
                stderr_text = result.stderr.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                stderr_text = result.stderr.decode("gbk", errors="replace")
            parts.append(f"[STDERR]\n{stderr_text}")
        
        # 检测是否删除了记忆文件，是则更新 MEMORY.md
        memory_dir = get_memory_dir()
        command_lower = command.lower()
        delete_patterns = [r"\brm\s", r"\bdel\s", r"\bRemove-Item\s", r"\brmdir\s"]
        is_delete = any(re.search(pattern, command_lower) for pattern in delete_patterns)
        if is_delete:
            # 检查命令是否包含记忆目录路径
            if str(memory_dir).lower() in command_lower:
                update_memory_index()
        
        return "\n".join(parts) if parts else "(无输出)"
    except subprocess.TimeoutExpired:
        return f"错误：命令执行超时 ({timeout_sec:.1f}秒)"
    except OSError as e:
        return f"错误：命令执行失败: {e}"


def _web_fetch(arguments: dict) -> str:
    """
    获取 URL 内容并以文本返回
    
    对于 HTML 页面会去除 script、style 和标签，返回可读文本。
    JSON/文本响应直接返回。
    
    参数:
        arguments: 包含 url 和可选 max_length 的字典
        
    返回:
        网页内容，或错误信息
    """
    url = arguments.get("url")
    max_length = arguments.get("max_length", 50000)
    if not url:
        return "错误：缺少必需参数 url"
    try:
        parsed = urllib.parse.urlparse(url)
        encoded_path = urllib.parse.quote(parsed.path, safe="/:")
        encoded_query = urllib.parse.quote(parsed.query, safe="&=")
        encoded_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, encoded_path, parsed.params, encoded_query, parsed.fragment))
        req = urllib.request.Request(encoded_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text" not in content_type and "json" not in content_type and "xml" not in content_type:
                return f"错误：Content-Type ({content_type}) 不是文本类型"
            charset = response.headers.get_content_charset()
            content = response.read()
            if charset:
                content = content.decode(charset, errors="replace")
            else:
                content = content.decode("utf-8", errors="replace")
        content = content[:max_length]
        try:
            content = html_module.unescape(content)
        except (html_module.HTMLParseError, ValueError):
            pass
        content = _RE_SCRIPT.sub("", content)
        content = _RE_STYLE.sub("", content)
        content = _RE_TAG.sub("", content)
        content = _RE_WHITESPACE.sub(" ", content).strip()
        return content if content else "(无内容)"
    except urllib.error.URLError as e:
        return f"错误：网络请求失败: {e}"
    except TimeoutError:
        return "错误：请求超时"
    except OSError as e:
        return f"错误：获取网页失败: {e}"


# 工具实现映射表（agent 工具在 agent.py 中单独处理）
_TOOL_IMPLEMENTATIONS = {
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "list_files": _list_files,
    "grep_search": _grep_search,
    "run_shell": _run_shell,
    "web_fetch": _web_fetch,
}


# 危险命令模式（用于权限检查）
DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE),
    re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE),
    re.compile(r"\bStop-Process\s", re.IGNORECASE),
]


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    执行工具调用
    
    参数:
        tool_name: 工具名称
        arguments: 工具参数字典
        
    返回:
        工具执行结果字符串
    """
    impl = _TOOL_IMPLEMENTATIONS.get(tool_name)
    if not impl:
        return f"错误：未知工具: {tool_name}"
    return impl(arguments)


def get_tools_for_agent_type(agent_type: str) -> list[dict]:
    """
    根据 agent 类型返回对应的工具集
    
    参数:
        agent_type: 子 Agent 类型（"explore" | "plan" | "general"）
        
    返回:
        对应类型可用的工具定义列表
    """
    if agent_type in ["explore", "plan"]:
        # explore/plan: 仅本地代码工具 (read_file, list_files, grep_search)
        read_only_tools = ["read_file", "list_files", "grep_search"]
        return [tool for tool in tool_definitions if tool["function"]["name"] in read_only_tools]
    elif agent_type == "general":
        # general: 除 agent 外的所有工具（防止递归）
        return [tool for tool in tool_definitions if tool["function"]["name"] != "agent"]
    else:
        # 默认返回所有工具
        return tool_definitions


def check_permission(tool_name: str, arguments: dict) -> tuple[bool, str]:
    """
    检查工具调用是否需要用户确认（危险命令检测）
    
    参数:
        tool_name: 工具名称
        arguments: 工具参数字典
        
    返回:
        (是否需要确认, 警告信息) 元组
    """
    if tool_name != "run_shell":
        return (False, "")
    command = arguments.get("command", "")
    command_lower = command.lower()
    
    # 检查是否操作 .codeagent 目录
    codeagent_path = Path(__file__).parent / ".codeagent"
    if str(codeagent_path).lower() in command_lower:
        return (False, "")
    
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return (True, f"危险命令检测: {pattern.pattern} 匹配到命令: {command[:100]}")
    return (False, "")
