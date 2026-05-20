"""
ui.py - 终端用户界面模块
========================

本模块提供与用户交互的终端显示功能，使用 Rich 库实现美化的输出。
主要功能包括：
- 欢迎信息和用户提示
- 助手和子 Agent 输出显示
- 工具调用显示
- 费用统计显示
- 危险操作确认
- 记忆内容显示
"""

from rich.console import Console
import sys

# 全局控制台实例（禁用代码高亮以避免误解析）
console = Console(highlight=False)


def print_welcome() -> None:
    """打印欢迎信息和可用命令"""
    console.print("\n[bold cyan]CodeAgent助手[/bold cyan]\n")
    console.print("[dim] 命令: /exit /clear /cost /compact /memory [/dim]\n")


def print_user_prompt() -> None:
    """打印用户输入提示符"""
    console.print("\n[bold green]> [/bold green]", end="")


def print_divider() -> None:
    """打印分隔线"""
    console.print(f"\n[dim]{'─' * 50}[/dim]")


def print_error(msg: str) -> None:
    """
    打印错误信息
    
    参数:
        msg: 错误信息字符串
    """
    console.print(f"\n [red]错误：{msg}[/red]")


def print_assistant_start() -> None:
    """打印助手输出的前缀"""
    console.print("\n[bold blue]助手:[/bold blue] ", end="")


def print_subagent_start(agent_type: str) -> None:
    """
    打印子 Agent 启动信息
    
    参数:
        agent_type: 子 Agent 类型（"explore" | "plan" | "general"）
    """
    agent_type_names = {
        "explore": "检索",
        "plan": "规划",
        "general": "通用",
    }
    name = agent_type_names.get(agent_type, "通用")
    console.print(f"\n[bold purple]🚀 子Agent({name}):[/bold purple] ", end="")


def print_subagent_content(content: str) -> None:
    """
    流式打印子 Agent 输出内容
    
    参数:
        content: 输出文本
    """
    sys.stdout.write(content)
    sys.stdout.flush()


def print_abort() -> None:
    """打印用户取消操作的提示"""
    console.print("\n[yellow]用户取消[/yellow]")


def print_assistant_content(content: str) -> None:
    """
    流式打印助手输出内容
    
    参数:
        content: 输出文本
    """
    sys.stdout.write(content)
    sys.stdout.flush()


def print_message(text: str, style: str = "") -> None:
    """
    通用打印函数，传入文本和颜色样式
    
    参数:
        text: 要打印的文本
        style: Rich 样式字符串（可选）
    """
    if style:
        console.print(f"[{style}]{text}[/{style}]")
    else:
        console.print(text)


def _calculate_cost(usage: dict) -> tuple[float, float]:
    """
    计算成本和缓存比例（内部函数）
    
    定价：
    - 非缓存输入：¥1.0 / 百万 tokens
    - 缓存输入：¥0.02 / 百万 tokens
    - 输出：¥2.0 / 百万 tokens
    
    参数:
        usage: 包含 input_tokens, output_tokens, cached_tokens 的字典
        
    返回:
        (成本金额, 缓存比例%) 元组
    """
    cached_ratio = (usage['cached_tokens'] / usage['input_tokens'] * 100) if usage['input_tokens'] > 0 else 0.0
    cost = (usage['input_tokens'] * 1.0 + usage['output_tokens'] * 2.0) / 1_000_000 - (usage['cached_tokens'] * 0.98 / 1_000_000)
    return cost, cached_ratio


def print_total_usage(usage: dict) -> None:
    """
    打印简洁的累计 Tokens 统计
    
    参数:
        usage: 消费统计字典
    """
    cost, cached_ratio = _calculate_cost(usage)
    console.print(f"[dim]累计 Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out (¥{cost:.4f})[/dim]")


def print_memory_content(content: str) -> None:
    """
    打印 MEMORY.md 内容
    
    参数:
        content: 记忆文件内容
    """
    console.print(f"\n[bold cyan]MEMORY.md 内容:[/bold cyan]")
    console.print(content)


# 定价表（每百万 tokens，单位：元）
PRICING_PER_MILLION = {
    "input_cached": 0.02,
    "input_not_cached": 1.0,
    "output": 2.0,
}


def print_billing(input_tokens: int, output_tokens: int, cached_tokens: int, total_usage: dict) -> None:
    """
    打印本次和累计的费用统计
    
    参数:
        input_tokens: 本次输入 tokens
        output_tokens: 本次输出 tokens
        cached_tokens: 本次缓存 tokens
        total_usage: 累计使用统计字典
    """
    # current_usage = {
    #     "input_tokens": input_tokens,
    #     "output_tokens": output_tokens,
    #     "cached_tokens": cached_tokens,
    # }
    #cost, cached_ratio = _calculate_cost(current_usage)
    #console.print(f"\n[dim]本次 Tokens: {input_tokens} in / {output_tokens} out (¥{cost:.4f})[/dim]")
    console.print(f"\n")
    print_total_usage(total_usage)


# ─── 工具显示相关 ─────────────────────────────────────────

# 工具图标映射
_TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "🔧",
    "list_files": "📁",
    "grep_search": "🔍",
    "run_shell": "💻",
    "agent": "🚀",
}


def _get_tool_icon(name: str) -> str:
    """
    获取工具对应的图标
    
    参数:
        name: 工具名称
        
    返回:
        图标字符串
    """
    return _TOOL_ICONS.get(name, "🔨")


# 工具结果默认最大显示长度
_tool_result_max_length: int = 300


def print_tool_start(name: str, arguments: dict) -> None:
    """
    打印工具调用开始信息
    
    参数:
        name: 工具名称
        arguments: 工具参数字典
    """
    icon = _get_tool_icon(name)
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
    console.print(f"\n{icon} [bold cyan]{name}[/bold cyan]({args_str})")


def print_tool_result(result: str, max_length: int | None = None) -> None:
    """
    打印工具调用结果
    
    参数:
        result: 工具结果字符串
        max_length: 最大显示长度（可选）
    """
    if max_length is None:
        max_length = _tool_result_max_length
    console.print(f"[dim]{'─' * 30}[/dim]")
    if len(result) > max_length:
        console.print(result[:max_length] + f"\n[dim]... (共 {len(result)} 字符)[/dim]")
    else:
        console.print(result)


def set_tool_result_max_length(length: int) -> None:
    """
    设置工具结果最大显示长度
    
    参数:
        length: 最大长度
    """
    global _tool_result_max_length
    _tool_result_max_length = length


def ask_dangerous_confirmation(reason: str) -> bool:
    """
    询问用户是否确认执行危险操作
    
    参数:
        reason: 危险原因说明
        
    返回:
        True 表示用户确认执行，False 表示取消
    """
    console.print(f"\n[bold yellow]⚠️ 危险操作警告[/bold yellow]")
    console.print(f"[yellow]{reason}[/yellow]")
    console.print("[yellow]是否继续执行？输入 'y' 确认，或 'n' 取消[/yellow] ")
    while True:
        response = input("> ").strip().lower()
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            console.print("[yellow]请输入 'y' 确认或 'n' 取消[/yellow] ")
