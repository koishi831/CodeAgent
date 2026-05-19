"""
__main__.py - 主程序入口模块
============================

本模块负责：
1. 加载环境配置
2. 初始化 Agent 实例
3. 运行交互式 REPL 循环
4. 处理特殊命令（/exit, /clear, /cost, /compact, /memory）

主要流程：
load_env() -> main() -> Agent() -> run_repl()
"""

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
    """
    加载环境变量配置
    
    从以下位置按优先级加载 .env 文件：
    1. 当前工作目录的 .env
    2. 本模块所在目录的 .env
    
    配置项：
    - OPENAI_MODEL_ID: 模型名称（如 deepseek-v4, glm-4-flash 等）
    - OPENAI_BASE_URL: API 基础地址
    - OPENAI_API_KEY: API 密钥
    """
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break


def run_repl(agent: Agent) -> None:
    """
    运行交互式 REPL (Read-Eval-Print Loop) 循环
    
    功能：
    - 持续接收用户输入
    - 处理特殊命令
    - 调用 Agent 进行对话
    - 处理中断信号（Ctrl+C）
    
    特殊命令：
    - /exit: 退出程序
    - /clear: 清空对话历史
    - /cost: 显示总 token 消耗
    - /compact: 手动压缩上下文
    - /memory: 显示记忆内容
    
    参数:
        agent: Agent 实例
    """
    # 设置 Ctrl+C 信号处理器，用于中断当前操作
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

        # 正常对话处理
        try:
            agent.chat(content)
        except Exception as e:
            print_error(str(e))
            continue
        print_divider()


def main() -> None:
    """
    主函数 - 程序入口
    
    流程：
    1. 加载环境变量
    2. 从环境变量获取配置
    3. 创建 Agent 实例
    4. 启动 REPL 循环
    
    配置项：
    - model: 模型名称，来自 OPENAI_MODEL_ID
    - api_base: API 地址，来自 OPENAI_BASE_URL
    - api_key: API 密钥，来自 OPENAI_API_KEY
    - thinking: DeepSeek 思考模式，默认为 "enabled"
    - reasoning_effort: 思考强度，默认为 "high"
    """
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
