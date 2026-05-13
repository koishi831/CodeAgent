
from pathlib import Path
import os
import signal

from dotenv import load_dotenv

from .agent import Agent
from .ui import print_abort, print_divider, print_error, print_user_prompt, print_welcome


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
            print("\n\n退出程序")
            break
        content = content.strip()
        if content == "exit":
            print("\n\n退出程序")
            break
        if not content:
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
    )
    run_repl(agent)

if __name__ == "__main__":
    main()