
from pathlib import Path
import os
import signal

from dotenv import load_dotenv

from .agent import Agent
from .memory import get_memory_dir
from .ui import (
    print_abort, print_divider, print_error, print_user_prompt, print_welcome,
    print_message, print_total_usage, print_memory_content
)


def load_env():
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break


def run_repl(agent: Agent) -> None:
    def _sigint_handler(signum, frame):
        agent.abort()

    signal.signal(signal.SIGINT, _sigint_handler)

    print_welcome()
    while True:
        print_user_prompt()
        try:
            content = input()
        except (KeyboardInterrupt, EOFError):
            print("退出程序\n")
            break
        content = content.strip()
        if not content:
            continue

        # 处理 REPL 命令
        if content == "/exit":
            print("退出程序\n")
            break
        elif content == "/clear":
            agent.clear_history()
            print_message("\n对话历史已清空", style="green")
            print_divider()
            continue
        elif content == "/cost":
            usage = agent.get_total_usage()
            print_total_usage(usage)
            print_divider()
            continue
        elif content == "/compact":
            messages, token_count = agent.compact_context()
            if messages is None:
                print_message("\n对话太短，无需压缩", style="yellow")
            else:
                print_message(f"\n对话已压缩，当前 Token: {token_count}", style="green")
            print_divider()
            continue
        elif content == "/memory":
            memory_path = get_memory_dir() / "MEMORY.md"
            if memory_path.exists():
                with open(memory_path, 'r', encoding='utf-8') as f:
                    print_memory_content(f.read())
            else:
                print_message("\n暂无记忆", style="yellow")
            print_divider()
            continue

        try:
            agent.chat(content)
        except Exception as e:
            print_error(str(e))
            continue
        print_divider()


def main() -> None:
    load_env()

    agent = Agent(
        model=os.environ.get("OPENAI_MODEL_ID"),
        api_base=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        thinking="enabled",  # DeepSeek 思考模式：enabled/disabled
        reasoning_effort="high",  # 思考强度：high/max
    )
    run_repl(agent)

if __name__ == "__main__":
    main()

