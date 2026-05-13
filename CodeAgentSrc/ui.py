
from rich.console import Console
import sys

# 全局控制台实例（禁用代码高亮）
console = Console(highlight=False)

def print_welcome() -> None:
    console.print("\n[bold cyan]CodeAgent助手[/bold cyan]\n")
    console.print("[dim]'exit' 退出[/dim]")


def print_user_prompt() -> None:
    console.print("\n[bold green]> [/bold green]", end="")

def print_divider() -> None:
    console.print(f"\n[dim]{'─' * 50}[/dim]")

def print_error(msg: str) -> None:
    console.print(f"\n [red]错误：{msg}[/red]")

def print_assistant_start() -> None:
    console.print("\n[bold blue]助手:[/bold blue] ", end="")

def print_abort() -> None:
    console.print("\n[yellow]用户取消[/yellow]")
        
def print_assistant_content(content: str) -> None:
    sys.stdout.write(content)
    sys.stdout.flush()

PRICING_PER_MILLION = {
    "input_cached": 0.02,
    "input_not_cached": 1.0,
    "output": 2.0,
}

def print_billing(input_tokens: int, output_tokens: int, cached_tokens: int, total_usage: dict) -> None:
    cached_ratio = (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0.0
    cost = (input_tokens * 1.0 + output_tokens * 2.0) / 1_000_000 - (cached_tokens * 0.98 / 1_000_000)
    total_cost = (total_usage["input_tokens"] * 1.0 + total_usage["output_tokens"] * 2.0) / 1_000_000 - (total_usage["cached_tokens"] * 0.98 / 1_000_000)
    total_cached_ratio = (total_usage["cached_tokens"] / total_usage["input_tokens"] * 100) if total_usage["input_tokens"] > 0 else 0.0
    console.print(f"\n[dim]本次 Tokens: {input_tokens} in / {output_tokens} out , cache {cached_ratio:.1f}%(¥{cost:.4f})[/dim]")
    console.print(f"[dim]累计 Tokens: {total_usage['input_tokens']} in / {total_usage['output_tokens']} out , cache {total_cached_ratio:.1f}%(¥{total_cost:.4f})[/dim]")

