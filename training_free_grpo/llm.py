import time
import openai
from utu.utils import EnvUtils
from google import genai
import re

class LLM:
    def __init__(self):
        EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL", "UTU_LLM_BASE_URL", "UTU_LLM_API_KEY"])
        self.model_name = EnvUtils.get_env("UTU_LLM_MODEL")
        self.base_url = EnvUtils.get_env("UTU_LLM_BASE_URL")
        self.api_key = EnvUtils.get_env("UTU_LLM_API_KEY")
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(self, messages_or_prompt, max_tokens=4096, temperature=0, max_retries=3, return_reasoning=False):
        for attempt in range(max_retries):
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError("messages_or_prompt must be a string or a list of messages.")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=120.0,
                    extra_body={
                        "reasoning_effort": "none",
                        "reasoning": {"effort": "none"},
                    },
                )

                message = response.choices[0].message
                raw_text = message.content

                if raw_text is None:
                    raise RuntimeError(f"LLM returned empty content. Full response: {response}")

                response_text = raw_text.strip()
                reasoning = getattr(message, "reasoning_content", None)

                if return_reasoning:
                    return response_text, reasoning

                return response_text

            except Exception as e:
                last_error = e
                print(f"LLM call failed on attempt {attempt + 1}/{max_retries}: {repr(e)}")
                time.sleep(3)

        raise RuntimeError(f"LLM call failed after {max_retries} retries") from last_error