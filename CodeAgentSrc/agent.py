
import json
import time
import errno
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from openai import OpenAI
from .ui import print_abort, print_divider, print_assistant_start, print_assistant_content, print_billing, print_tool_start, print_tool_result, ask_dangerous_confirmation, print_subagent_start
from .tool import execute_tool, tool_definitions, check_permission, get_tools_for_agent_type
from .prompt import build_system_prompt
from .subagent import build_agent_descriptions, get_prompt_for_agent_type

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

def _estimate_single_message(msg: Dict[str, Any]) -> int:
    tokens = 4  # 系统提示词+角色
    if msg.get("content"):
        content = msg["content"]
        tokens += int(len(content) * 0.8)
    if msg.get("reasoning_content"):
        reasoning = msg["reasoning_content"]
        tokens += int(len(reasoning) * 0.8)
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            tokens += 8
            tokens += int(len(tc.get("function", {}).get("name", "")) * 0.5)
            tokens += int(len(tc.get("function", {}).get("arguments", "")) * 0.7)
    if msg.get("tool_call_id"):
        tokens += 6
    return tokens

def _estimate_tokens(messages: List[Dict[str, Any]], system_prompt_tokens: Optional[int] = None) -> int:
    tokens = 0
    for msg in messages:
        if system_prompt_tokens is not None and msg.get("role") == "system":
            tokens += system_prompt_tokens
            continue
        tokens += _estimate_single_message(msg)
    return tokens

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

def _retry_with_backoff(func: Callable[..., Any], *args: Any, max_retries: int = 3, **kwargs: Any) -> Any:
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
    def __init__(self, model: str, api_base: str, api_key: str, thinking: str = "enabled", reasoning_effort: str = "high"):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()}
        ]
        self.effective_window = _get_context_window(model)
        self.total_usage: Dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }
        self._interrupted = False
        # 缓存系统提示词token，系统提示词不会压缩，不用每次都算
        self._system_prompt_tokens = _estimate_single_message(self.messages[0])
        self._last_token_count = _estimate_tokens(self.messages, self._system_prompt_tokens)

    def clear_history(self) -> None:
        """清空对话历史，保留系统提示词"""
        self.messages = [{"role": "system", "content": build_system_prompt()}]
        self._interrupted = False
        self._last_token_count = _estimate_tokens(self.messages, self._system_prompt_tokens)

    def get_total_usage(self) -> Dict[str, int]:
        """获取总消费 token 统计"""
        return self.total_usage.copy()

    def compact_context(self) -> Tuple[Optional[List[Dict[str, Any]]], int]:
        """压缩对话上下文"""
        if len(self.messages) < 5:
            return None, _estimate_tokens(self.messages, self._system_prompt_tokens)
        self.messages, _ = self._manage_context(self.messages, force_compress=True)
        self._last_token_count = _estimate_tokens(self.messages, self._system_prompt_tokens)
        return self.messages, self._last_token_count

    def abort(self) -> None:
        self._interrupted = True

    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """统一工具调用分发"""
        if tool_name == "agent":
            return self._execute_agent_tool(arguments)
        else:
            return execute_tool(tool_name, arguments)

    def _execute_agent_tool(self, arguments: Dict[str, Any]) -> str:
        """运行子 agent"""
        prompt = arguments.get("prompt", "")
        agent_type = arguments.get("type", "general")
        
        if not prompt:
            return "错误：缺少必需参数 prompt"
        
        print_subagent_start(agent_type)
        
        # 使用与主 agent 相同的配置
        model = self.model
        api_base = None
        api_key = None
        
        # 从环境变量读取配置作为备用
        if not model:
            model = os.environ.get("OPENAI_MODEL_ID")
        api_base = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not model or not api_base or not api_key:
            return "错误：子 agent 配置缺失，请检查环境变量 OPENAI_MODEL_ID、OPENAI_BASE_URL、OPENAI_API_KEY"

        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        system_prompt = get_prompt_for_agent_type(agent_type)
        tools = get_tools_for_agent_type(agent_type)
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        effective_window = _get_context_window(model)
        system_prompt_tokens = _estimate_tokens([{"role": "system", "content": system_prompt}])
        
        result = ""
        subagent_total_input = 0
        subagent_total_output = 0
        subagent_total_cached = 0
        
        def call_api() -> Any:
            current_tokens = _estimate_tokens(messages, system_prompt_tokens)
            # 子 agent 使用简化的上下文管理
            trimmed_messages, _ = self._trim_context_simple(messages, system_prompt_tokens)
            
            api_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": trimmed_messages,
                "tools": tools,
                "stream": False
            }
            
            # 使用与主 agent 一致的推理配置
            if "deepseek" in model.lower():
                api_kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
                api_kwargs["reasoning_effort"] = self.reasoning_effort
            
            return client.chat.completions.create(**api_kwargs)
        
        while True:
            if self._interrupted:
                print_abort()
                break
                
            response = _retry_with_backoff(call_api)
            message = response.choices[0].message
            
            # 统计 token 消耗
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
                # 尝试多种方式获取 cached tokens
                cached_tokens = 0
                try:
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        if hasattr(response.usage.prompt_tokens_details, 'cache_read'):
                            cached_tokens = response.usage.prompt_tokens_details.cache_read or 0
                except:
                    pass
                subagent_total_input += input_tokens
                subagent_total_output += output_tokens
                subagent_total_cached += cached_tokens
            
            if message.content:
                result += message.content
            
            tool_calls_dict = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls_dict.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": message.content if message.content else None,
                "tool_calls": tool_calls_dict if tool_calls_dict else None
            }
            
            # 如果有 reasoning_content，必须保存下来（DeepSeek 要求回传）
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                assistant_msg["reasoning_content"] = message.reasoning_content
            
            messages.append(assistant_msg)
            
            if not message.tool_calls:
                break
            
            for tool_call in message.tool_calls:
                if self._interrupted:
                    print_abort()
                    break
                    
                try:
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                
                print_tool_start(tool_call.function.name, args)
                
                need_confirm, reason = check_permission(tool_call.function.name, args)
                if need_confirm:
                    # 子 agent 不进行危险操作，直接拒绝
                    tool_result = "危险操作被拒绝"
                else:
                    tool_result = _retry_with_backoff(execute_tool, tool_call.function.name, args)
                
                print_tool_result(tool_result)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            if self._interrupted:
                break
        
        # 把子 agent 的 token 消耗累加到主 agent
        self.total_usage["input_tokens"] += subagent_total_input
        self.total_usage["output_tokens"] += subagent_total_output
        self.total_usage["cached_tokens"] += subagent_total_cached
        
        return result or "子 agent 执行完成"

    def _trim_context_simple(self, messages: List[Dict[str, Any]], system_prompt_tokens: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
        """简化的上下文裁剪，用于子 agent"""
        result = []
        # 保留系统提示词
        for msg in messages:
            if msg.get("role") == "system":
                result.append(msg)
                break
        if not result:
            result = []

        # 跟踪工具调用和结果
        tool_call_map: Dict[str, Dict[str, Any]] = {}
        tool_result_map: Dict[str, Dict[str, Any]] = {}
        read_file_map: Dict[str, str] = {}
        grep_search_ids: List[str] = []
        list_files_ids: List[str] = []
        run_shell_ids: List[str] = []
        recent_tool_ids: List[str] = []

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
                filtered_tool_calls = [tc for tc in msg["tool_calls"] if tc["id"] in keep_ids]
                if filtered_tool_calls:
                    filtered_msg = msg.copy()
                    filtered_msg["tool_calls"] = filtered_tool_calls
                    if "reasoning_content" in msg:
                        filtered_msg["reasoning_content"] = msg["reasoning_content"]
                    final_result.append(filtered_msg)
                    i += 1
                else:
                    i += 1
            elif msg.get("role") == "tool":
                if msg["tool_call_id"] in keep_ids:
                    final_result.append(msg)
                i += 1
            else:
                final_result.append(msg)
                i += 1

        token_count = _estimate_tokens(final_result, system_prompt_tokens)
        return final_result, token_count

    def _manage_context(self, messages: List[Dict[str, Any]], force_compress: bool = False) -> Tuple[List[Dict[str, Any]], int]:
        """统一的上下文管理方法：先裁剪，超过阈值再压缩"""
        # 第一步：裁剪上下文
        trimmed, token_count = self._trim_context_simple(messages, self._system_prompt_tokens)
        
        # 如果不需要强制压缩且 token 数在阈值内，直接返回
        if not force_compress and token_count <= self.effective_window * 0.5:
            return trimmed, token_count
        
        # 第二步：压缩上下文（使用摘要）
        if len(trimmed) < 5:
            return trimmed, token_count
        
        # 找到系统提示词
        system_msg = None
        for msg in trimmed:
            if msg.get("role") == "system":
                system_msg = msg
                break
        
        # 找到最新用户消息
        last_user_msg = None
        for i in range(len(trimmed) - 1, -1, -1):
            if trimmed[i].get("role") == "user":
                last_user_msg = trimmed[i]
                break
        
        if not system_msg or not last_user_msg:
            return trimmed, token_count
        
        # 找到至少2轮对话（包含最新用户消息）
        dialog_history: List[Dict[str, Any]] = []
        user_count = 0
        i = len(trimmed) - 1
        while i >= 0 and user_count < 2:
            msg = trimmed[i]
            dialog_history.insert(0, msg)
            if msg.get("role") == "user":
                user_count += 1
            i -= 1
        
        # 准备待压缩的历史（去掉系统提示词和dialog_history）
        history_to_compress: List[Dict[str, Any]] = []
        i = 0
        while i < len(trimmed):
            msg = trimmed[i]
            if msg.get("role") == "system":
                i += 1
                continue
            found = False
            for dm in dialog_history:
                if dm == msg:
                    found = True
                    break
            if not found:
                history_to_compress.append(msg)
            i += 1
        
        if not history_to_compress:
            return trimmed, token_count
        
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
            
            # 统计压缩对话的 token 消耗
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
                # 尝试多种方式获取 cached tokens
                cached_tokens = 0
                try:
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        if hasattr(response.usage.prompt_tokens_details, 'cache_read'):
                            cached_tokens = response.usage.prompt_tokens_details.cache_read or 0
                except:
                    pass
                self.total_usage["input_tokens"] += input_tokens
                self.total_usage["output_tokens"] += output_tokens
                self.total_usage["cached_tokens"] += cached_tokens
        except Exception as e:
            summary = "[历史对话摘要失败，保留原对话]"
        
        # 构建新消息列表
        new_messages: List[Dict[str, Any]] = [system_msg]
        
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

        # 保存调用前的累计消耗，用于计算本次消耗
        prev_total_input = self.total_usage["input_tokens"]
        prev_total_output = self.total_usage["output_tokens"]
        prev_total_cached = self.total_usage["cached_tokens"]

        # 估算当前token数并管理上下文
        current_tokens = _estimate_tokens(self.messages, self._system_prompt_tokens)
        if current_tokens > self.effective_window * 0.4:
            trimmed_messages, self._last_token_count = self._manage_context(self.messages)
        else:
            trimmed_messages = self.messages
            self._last_token_count = current_tokens

        # 构建 API 调用参数
        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": trimmed_messages,
            "tools": tool_definitions,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        # 如果是 DeepSeek 模型，添加思考模式配置
        if "deepseek" in self.model.lower():
            api_kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
            api_kwargs["reasoning_effort"] = self.reasoning_effort
        
        response = self.client.chat.completions.create(**api_kwargs)

        print_assistant_start()
        assistant_message = ""
        reasoning_content = ""  # 新增：保存思考内容
        tool_calls: List[Dict[str, Any]] = []
        current_tool_call: Optional[Dict[str, Any]] = None

        for chunk in response:
            if self._interrupted:
                print_abort()
                return
            # 处理 reasoning_content (思考内容)
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
                # 可以选择打印思考内容，这里暂时不打印
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
                chunk_input = chunk.usage.prompt_tokens or 0
                chunk_output = chunk.usage.completion_tokens or 0
                # 尝试多种方式获取 cached tokens
                chunk_cached = 0
                try:
                    if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details:
                        if hasattr(chunk.usage.prompt_tokens_details, 'cache_read'):
                            chunk_cached = chunk.usage.prompt_tokens_details.cache_read or 0
                except:
                    pass
                # 使用实际的input_tokens更新记录
                if chunk_input > 0:
                    self._last_token_count = chunk_input
                # 及时累加到总消耗
                self.total_usage["input_tokens"] += chunk_input
                self.total_usage["output_tokens"] += chunk_output
                self.total_usage["cached_tokens"] += chunk_cached

        if tool_calls:
            assistant_msg: Dict[str, Any] = {
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
            }
            # 如果有 reasoning_content，必须保存下来（DeepSeek 要求回传）
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            self.messages.append(assistant_msg)
            for tc in tool_calls:
                if self._interrupted:
                    print_abort()
                    break
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
                try:
                    tool_result = _retry_with_backoff(self._execute_tool_call, tc["function"]["name"], args)
                except Exception as e:
                    tool_result = f"工具执行失败: {e}"
                print_tool_result(tool_result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result
                })
            # 继续对话，发送工具结果给模型
            self.chat("")
            return

        # 保存没有 tool calls 的消息，也需要包含 reasoning_content
        final_assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message
        }
        if reasoning_content:
            final_assistant_msg["reasoning_content"] = reasoning_content
        self.messages.append(final_assistant_msg)

        # 计算本次消耗 = 当前累计 - 调用前累计
        this_input = self.total_usage["input_tokens"] - prev_total_input
        this_output = self.total_usage["output_tokens"] - prev_total_output
        this_cached = self.total_usage["cached_tokens"] - prev_total_cached

        print_billing(this_input, this_output, this_cached, self.total_usage)

