
from openai import OpenAI
from .ui import print_abort, print_divider, print_assistant_start, print_assistant_content, print_billing

MODEL_CONTEXT = {
    "glm-4-flash": 128000,
    "glm-4.5-air": 128000,
    "deepseek-v4-flash": 1000000,
}

def _get_context_window(model: str) -> int:
    return MODEL_CONTEXT.get(model, 200000)


class Agent:
    def __init__(self, model: str, api_base: str, api_key: str):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.effective_window = _get_context_window(model)
        self.messages = []
        self.total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }
        self._interrupted = False

    def abort(self) -> None:
        self._interrupted = True

    def chat(self, user_message: str) -> None:
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        self._interrupted = False

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )

        print_assistant_start()
        assistant_message = ""
        output_tokens = 0
        input_tokens = 0
        cached_tokens = 0

        for chunk in response:
            if self._interrupted:
                print_abort()
                return
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assistant_message += content
                print_assistant_content(content)
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
                cached_tokens = (chunk.usage.prompt_tokens_details.cache_read if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details else 0) or 0

        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        self.total_usage["input_tokens"] += input_tokens
        self.total_usage["output_tokens"] += output_tokens
        self.total_usage["cached_tokens"] += cached_tokens

        print_billing(input_tokens, output_tokens, cached_tokens, self.total_usage)

