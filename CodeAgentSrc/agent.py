
import json
import time
import errno
from openai import OpenAI
from .ui import print_abort, print_divider, print_assistant_start, print_assistant_content, print_billing, print_tool_start, print_tool_result, ask_dangerous_confirmation
from .tool import execute_tool, tool_definitions, check_permission
from .prompt import build_system_prompt

# 模型上下文窗口大小
MODEL_CONTEXT = {
    "glm-4-flash": 128000,
    "glm-4.5-air": 128000,
    "deepseek-v4-flash": 1000000,
}

def _get_context_window(model: str) -> int:
    return MODEL_CONTEXT.get(model, 200000)

# 重试策略
RETRYABLE_HTTP_CODES = {429, 503, 529}
RETRYABLE_ERRORS = {"overloaded", "ECONNRESET", "ETIMEDOUT"}

def _is_retryable_error(error: Exception) -> bool:
    error_str = str(error).lower()
    if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        if error.response.status_code in RETRYABLE_HTTP_CODES:
            return True
    for retryable in RETRYABLE_ERRORS:
        if retryable.lower() in error_str:
            return True
    if hasattr(error, 'errno'):
        if error.errno in (errno.ECONNRESET, errno.ETIMEDOUT):
            return True
    return False

def _retry_with_backoff(func, *args, max_retries=3, **kwargs):
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries and _is_retryable_error(e):
                wait_time = (2 ** attempt) + (time.time() % 1)
                time.sleep(wait_time)
                continue
            raise
    raise last_exception


class Agent:
    def __init__(self, model: str, api_base: str, api_key: str):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.messages = [
            {"role": "system", "content": build_system_prompt()}
        ]
        self.effective_window = _get_context_window(model)
        self.total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }
        self._interrupted = False

    def abort(self) -> None:
        self._interrupted = True

    def chat(self, user_message: str) -> None:
        if user_message:
            self.messages.append({
                "role": "user",
                "content": user_message
            })
        self._interrupted = False

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tool_definitions,
            stream=True
        )

        print_assistant_start()
        assistant_message = ""
        output_tokens = 0
        input_tokens = 0
        cached_tokens = 0
        tool_calls = []
        current_tool_call = None

        for chunk in response:
            if self._interrupted:
                print_abort()
                return
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assistant_message += content
                print_assistant_content(content)
            if chunk.choices[0].delta.tool_calls:
                for tool_call_delta in chunk.choices[0].delta.tool_calls:
                    index = tool_call_delta.index
                    if current_tool_call is None or current_tool_call.get("index") != index:
                        current_tool_call = {
                            "index": index,
                            "id": tool_call_delta.id,
                            "function": {
                                "name": tool_call_delta.function.name or "",
                                "arguments": tool_call_delta.function.arguments or ""
                            }
                        }
                        tool_calls.append(current_tool_call)
                    else:
                        current_tool_call["function"]["arguments"] += tool_call_delta.function.arguments or ""
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
                cached_tokens = (chunk.usage.prompt_tokens_details.cache_read if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details and hasattr(chunk.usage.prompt_tokens_details, 'cache_read') else 0) or 0

        if tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": assistant_message if assistant_message else None,
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                } for tc in tool_calls]
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                print_tool_start(tc["function"]["name"], args)
                need_confirm, reason = check_permission(tc["function"]["name"], args)
                if need_confirm:
                    if not ask_dangerous_confirmation(reason):
                        print_abort()
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "用户取消执行"
                        })
                        continue
                tool_result = _retry_with_backoff(execute_tool, tc["function"]["name"], args)
                print_tool_result(tool_result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result
                })
            self.chat("")  # Continue conversation with tool results
            return

        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        self.total_usage["input_tokens"] += input_tokens
        self.total_usage["output_tokens"] += output_tokens
        self.total_usage["cached_tokens"] += cached_tokens

        print_billing(input_tokens, output_tokens, cached_tokens, self.total_usage)

