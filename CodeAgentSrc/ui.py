
from rich.console import Console
import sys

# 全局控制台实例（禁用代码高亮）
console = Console(highlight=False)

def print_welcome() -> None:
    console.print("\n[bold cyan]CodeAgent助手[/bold cyan]\n")
    console.print("[dim] 命令: /exit /clear /cost /compact /memory [/dim]\n")


def print_user_prompt() -> None:
    console.print("\n[bold green]> [/bold green]", end="")

def print_divider() -> None:
    console.print(f"\n[dim]{'─' * 50}[/dim]")

def print_error(msg: str) -> None:
    console.print(f"\n [red]错误：{msg}[/red]")

def print_assistant_start() -> None:
    console.print("\n[bold blue]助手:[/bold blue] ", end="")

def print_subagent_start(agent_type: str) -> None:
    agent_type_names = {
        "explore": "检索",
        "plan": "规划",
        "general": "通用",
    }
    name = agent_type_names.get(agent_type, "通用")
    console.print(f"\n[bold purple]🚀 子Agent({name}):[/bold purple]")

def print_abort() -> None:
    console.print("\n[yellow]用户取消[/yellow]")
        
def print_assistant_content(content: str) -> None:
    sys.stdout.write(content)
    sys.stdout.flush()

def print_message(text: str, style: str = "") -> None:
    """通用打印函数，传入文本和颜色样式"""
    if style:
        console.print(f"[{style}]{text}[/{style}]")
    else:
        console.print(text)

def _calculate_cost(usage: dict) -> tuple[float, float]:
    """计算成本和缓存比例"""
    cached_ratio = (usage['cached_tokens'] / usage['input_tokens'] * 100) if usage['input_tokens'] > 0 else 0.0
    cost = (usage['input_tokens'] * 1.0 + usage['output_tokens'] * 2.0) / 1_000_000 - (usage['cached_tokens'] * 0.98 / 1_000_000)
    return cost, cached_ratio

def print_total_usage(usage: dict) -> None:
    """
    打印简洁的累计 Tokens
    
    Args:
        usage: 消费统计
    """
    cost, cached_ratio = _calculate_cost(usage)
    console.print(f"[dim]累计 Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out , cache {cached_ratio:.1f}%(¥{cost:.4f})[/dim]")

def print_memory_content(content: str) -> None:
    """打印 MEMORY.md 内容"""
    console.print(f"\n[bold cyan]MEMORY.md 内容:[/bold cyan]")
    console.print(content)

PRICING_PER_MILLION = {
    "input_cached": 0.02,
    "input_not_cached": 1.0,
    "output": 2.0,
}

def print_billing(input_tokens: int, output_tokens: int, cached_tokens: int, total_usage: dict) -> None:
    current_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }
    cost, cached_ratio = _calculate_cost(current_usage)
    console.print(f"\n[dim]本次 Tokens: {input_tokens} in / {output_tokens} out (¥{cost:.4f})[/dim]")
    print_total_usage(total_usage)


# ─── 工具图标和摘要 ─────────────────────────────────────────

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
    return _TOOL_ICONS.get(name, "🔨")

_tool_result_max_length: int = 300

def print_tool_start(name: str, arguments: dict) -> None:
    icon = _get_tool_icon(name)
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
    console.print(f"\n{icon} [bold cyan]{name}[/bold cyan]({args_str})")

def print_tool_result(result: str, max_length: int | None = None) -> None:
    if max_length is None:
        max_length = _tool_result_max_length
    console.print(f"[dim]{'─' * 30}[/dim]")
    if len(result) > max_length:
        console.print(result[:max_length] + f"\n[dim]... (共 {len(result)} 字符)[/dim]")
    else:
        console.print(result)

def set_tool_result_max_length(length: int) -> None:
    global _tool_result_max_length
    _tool_result_max_length = length

def ask_dangerous_confirmation(reason: str) -> bool:
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
    
