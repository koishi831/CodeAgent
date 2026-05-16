
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


def _estimate_single_message(msg: dict) -> int:
    tokens = 4  # 系统提示词+角色
    if msg.get("content"):
        content = msg["content"]
        tokens += int(len(content) * 0.8)
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            tokens += 8
            tokens += int(len(tc.get("function", {}).get("name", "")) * 0.5)
            tokens += int(len(tc.get("function", {}).get("arguments", "")) * 0.7)
    if msg.get("tool_call_id"):
        tokens += 6
    return tokens


def _estimate_tokens(messages: list, system_prompt_tokens: int = None) -> int:
    tokens = 0
    for msg in messages:
        if system_prompt_tokens is not None and msg.get("role") == "system":
            tokens += system_prompt_tokens
            continue
        tokens += _estimate_single_message(msg)
    return tokens


def _trim_context(messages: list, system_prompt_tokens: int = None) -> list:
    result = []
    # 保留系统提示词
    for msg in messages:
        if msg.get("role") == "system":
            result.append(msg)
            break
    if not result:
        result = []

    # 跟踪工具调用和结果
    tool_call_map = {}  # tool_call_id -> 对应的assistant消息
    tool_result_map = {}  # tool_call_id -> 对应的tool消息
    read_file_map = {}  # file_path -> 对应的(tool_call_id, index)
    grep_search_ids = []  # 所有grep_search的id，按时间顺序
    list_files_ids = []  # 所有list_files的id，按时间顺序
    run_shell_ids = []  # 所有run_shell的id，按时间顺序
    recent_tool_ids = []  # 最近3个tool_result的id

    i = len(result)
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                tc_id = tc["id"]
                tool_call_map[tc_id] = msg
                tool_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                if tool_name == "read_file" and "file_path" in args:
                    read_file_map[args["file_path"]] = tc_id
                elif tool_name == "grep_search":
                    grep_search_ids.append(tc_id)
                elif tool_name == "list_files":
                    list_files_ids.append(tc_id)
                elif tool_name == "run_shell":
                    run_shell_ids.append(tc_id)
            i += 1
        elif msg.get("role") == "tool":
            tc_id = msg["tool_call_id"]
            tool_result_map[tc_id] = msg
            recent_tool_ids.append(tc_id)
            i += 1
        else:
            i += 1

    # 规则3：最近3个tool_result要保留，此规则最优先
    keep_ids = set(recent_tool_ids[-3:])

    # 规则1：同一文件被read_file多次读取只保留最新一次
    for tc_id in read_file_map.values():
        if tc_id not in keep_ids:
            keep_ids.add(tc_id)

    # 规则2："grep_search", "list_files", "run_shell"工具的同类搜索结果保留最新3个
    keep_ids.update(grep_search_ids[-3:])
    keep_ids.update(list_files_ids[-3:])
    keep_ids.update(run_shell_ids[-3:])

    # 重新构建消息列表
    final_result = result.copy()
    i = len(result)
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            # 只保留需要的tool_call
            filtered_tool_calls = [tc for tc in msg["tool_calls"] if tc["id"] in keep_ids]
            if filtered_tool_calls:
                filtered_msg = msg.copy()
                filtered_msg["tool_calls"] = filtered_tool_calls
                final_result.append(filtered_msg)
                i += 1
            else:
                # 没有需要保留的tool_call，跳过此assistant消息
                i += 1
        elif msg.get("role") == "tool":
            if msg["tool_call_id"] in keep_ids:
                final_result.append(msg)
            i += 1
        else:
            # 其他消息保留
            final_result.append(msg)
            i += 1

    token_count = _estimate_tokens(final_result, system_prompt_tokens)
    return final_result, token_count


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
        # 缓存系统提示词token，系统提示词不会压缩，不用每次都算
        self._system_prompt_tokens = _estimate_single_message(self.messages[0])
        self._last_token_count = _estimate_tokens(self.messages, self._system_prompt_tokens)

    def abort(self) -> None:
        self._interrupted = True

    def _compress_context(self, messages: list) -> tuple:
        # 总消息小于5条时直接return
        if len(messages) < 5:
            return messages, _estimate_tokens(messages, self._system_prompt_tokens)
        
        # 找到系统提示词
        system_msg = None
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
                break
        
        # 找到最新用户消息
        last_user_msg = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_msg = messages[i]
                break
        
        if not system_msg or not last_user_msg:
            return messages, _estimate_tokens(messages, self._system_prompt_tokens)
        
        # 找到至少2轮对话（包含最新用户消息）
        dialog_history = []
        user_count = 0
        i = len(messages) - 1
        while i >= 0 and user_count < 2:
            msg = messages[i]
            dialog_history.insert(0, msg)
            if msg.get("role") == "user":
                user_count += 1
            i -= 1
        
        # 准备待压缩的历史（去掉系统提示词和dialog_history）
        history_to_compress = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "system":
                i += 1
                continue
            # 检查是否在dialog_history中
            found = False
            for dm in dialog_history:
                if dm == msg:
                    found = True
                    break
            if not found:
                history_to_compress.append(msg)
            i += 1
        
        if not history_to_compress:
            return messages, _estimate_tokens(messages, self._system_prompt_tokens)
        
        # 调用大模型总结历史
        summary_prompt = """请将以下对话历史总结为简洁的摘要，保持关键信息，删除冗余内容：

"""
        for msg in history_to_compress:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "tool":
                summary_prompt += f"[工具结果] {content}\n\n"
            elif role == "assistant":
                summary_prompt += f"助手: {content}\n\n"
            else:
                summary_prompt += f"用户: {content}\n\n"
        
        summary_prompt += "\n请用简洁的中文总结以上对话内容。"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个对话摘要助手，擅长用简洁的语言总结对话历史。"},
                    {"role": "user", "content": summary_prompt}
                ],
                stream=False
            )
            summary = response.choices[0].message.content
        except Exception as e:
            summary = "[历史对话摘要失败，保留原对话]"
        
        # 构建新消息列表
        new_messages = [system_msg]
        
        # 添加摘要消息（作为系统消息的补充）
        new_messages.append({
            "role": "system",
            "content": f"以下是对话历史摘要：\n{summary}"
        })
        
        # 添加至少2轮对话
        new_messages.extend(dialog_history)
        
        return new_messages, _estimate_tokens(new_messages, self._system_prompt_tokens)

    def chat(self, user_message: str) -> None:
        if user_message:
            self.messages.append({
                "role": "user",
                "content": user_message
            })
        self._interrupted = False

        # 估算当前token数
        current_tokens = _estimate_tokens(self.messages, self._system_prompt_tokens)
        trimmed_messages = self.messages
        # 当effective_window使用率大于40%时才调用_trim_context
        if current_tokens > self.effective_window * 0.4:
            trimmed_messages, self._last_token_count = _trim_context(self.messages, self._system_prompt_tokens)
            # 如果_trim_context后使用率还大于50%，调用_compress_context
            if self._last_token_count > self.effective_window * 0.5:
                trimmed_messages, self._last_token_count = self._compress_context(trimmed_messages)
        else:
            self._last_token_count = current_tokens

        response = self.client.chat.completions.create(
            model=self.model,
            messages=trimmed_messages,
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
                # 使用实际的input_tokens更新记录
                if input_tokens > 0:
                    self._last_token_count = input_tokens

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

