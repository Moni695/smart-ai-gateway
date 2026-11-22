"""
Providers and Reliable Client module.
Handles real OpenAI-compatible API calls (Groq, OpenAI) as well as a built-in
deterministic Mock Provider with simulated latency, realistic tokens, and error injection
for testing retries and automatic fallback.
"""

import time
import asyncio
import random
import re
from typing import Dict, Any, List, Optional
import httpx

from config import settings


class LLMProviderException(Exception):
    def __init__(self, message: str, status_code: int = 500, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class RateLimitException(LLMProviderException):
    def __init__(self, message: str = "Rate limit exceeded (429)"):
        super().__init__(message, status_code=429, retryable=True)


class TimeoutException(LLMProviderException):
    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, status_code=408, retryable=True)


class MockLLMProvider:
    """
    Deterministic Mock Provider that simulates realistic LLM responses,
    token counts, and model behavioral differences between cheap and strong tiers.
    Also supports error injection to test gateway retries and fallback.
    """
    def __init__(self):
        self.simulate_failures: Dict[str, str] = {}  # model -> error_type ("rate_limit", "timeout", "server_error")

    def set_simulation_error(self, model: str, error_type: Optional[str]):
        """Injects errors for testing resilience."""
        if error_type:
            self.simulate_failures[model] = error_type
        elif model in self.simulate_failures:
            del self.simulate_failures[model]

    async def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        # Check simulated failures first
        sim_err = self.simulate_failures.get(model)
        if sim_err == "rate_limit":
            raise RateLimitException("Simulated 429 Too Many Requests")
        elif sim_err == "timeout":
            raise TimeoutException("Simulated ReadTimeout")
        elif sim_err == "server_error":
            raise LLMProviderException("Simulated 500 Internal Server Error", status_code=500, retryable=True)

        # Simulate brief network latency (30ms - 80ms for cheap, 100ms - 200ms for strong)
        is_strong = "70b" in model or "strong" in model or "4o" in model
        sim_delay = random.uniform(0.08, 0.16) if is_strong else random.uniform(0.02, 0.05)
        await asyncio.sleep(sim_delay)

        # Combine text content from messages
        prompt_text = " ".join([m.get("content", "") for m in messages if isinstance(m.get("content"), str)])
        prompt_tokens = max(10, len(prompt_text.split()) * 2)

        # Determine response based on messages and model tier
        response_text, completion_tokens = self._mock_reasoning(messages, is_strong)

        return {
            "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def _mock_reasoning(self, messages: List[Dict[str, str]], is_strong: bool) -> tuple[str, int]:
        # Check if this is an Agent ReAct conversation
        is_agent_flow = any("Action:" in m.get("content", "") or "Thought:" in m.get("content", "") for m in messages)
        has_system_react = any("Action: the action to take" in m.get("content", "") for m in messages)

        if is_agent_flow or has_system_react:
            return self._mock_agent_step(messages, is_strong)

        # Standard non-agent completion flow
        prompt_text = " ".join([m.get("content", "") for m in messages if isinstance(m.get("content"), str)])
        lower = prompt_text.lower()

        # Direct Math
        if "calculate" in lower or "what is" in lower:
            math_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", prompt_text)
            if math_match:
                n1, op, n2 = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
                if op == "+": res = n1 + n2
                elif op == "-": res = n1 - n2
                elif op == "*": res = n1 * n2
                else: res = n1 / n2 if n2 != 0 else 0
                return f"The result of {n1} {op} {n2} is {res}.", 25

        # Code Refactor / Debugging
        if "def " in prompt_text or "function" in lower or "code" in lower:
            if is_strong:
                return "```python\ndef optimized_solution(data):\n    return [x * 2 for x in data if x % 2 == 0]\n```\nTime complexity reduced to O(n).", 55
            else:
                return "Here is some code:\ndef sol(d):\n    pass", 20

        # General factual / easy questions
        if "capital of france" in lower:
            return "The capital of France is Paris.", 12
        if "capital of japan" in lower:
            return "The capital of Japan is Tokyo.", 12
        if "hello" in lower or "hi" in lower:
            return "Hello! How can I assist you with your project today?", 15

        if is_strong:
            return "Detailed analysis: Based on a rigorous breakdown of the request, the primary considerations are architectural efficiency and correctness.", 45
        else:
            return "Here is the answer to your request.", 18

    def _mock_agent_step(self, messages: List[Dict[str, str]], is_strong: bool) -> tuple[str, int]:
        """Handles multi-step ReAct agent steps."""
        # Find original user question
        question = ""
        for m in messages:
            if m.get("role") == "user" and "Question:" in m.get("content", ""):
                question = m["content"].replace("Question:", "").strip()
                break
        if not question:
            # Fallback to last user message
            question = messages[-1].get("content", "")

        q_lower = question.lower()
        last_msg = messages[-1].get("content", "")

        # 1. Observation handling (after tool execution)
        if last_msg.startswith("Observation:"):
            obs = last_msg.replace("Observation:", "").strip()

            # Handle multi-step tasks requiring a second tool call
            if "calculate 10 percent" in q_lower and "45,000,000" in obs:
                if is_strong:
                    return 'Thought: I have the revenue ($45,000,000). Now I need to calculate 10% of it.\nAction: Calculator\nAction Input: {"expression": "45000000 * 0.10"}', 35
                else:
                    return f"Thought: I have the revenue.\nFinal Answer: {obs}", 25

            if "calculate 5 percent" in q_lower and "82,000,000" in obs:
                if is_strong:
                    return 'Thought: I have the revenue ($82,000,000). Now I need to calculate 5% of it.\nAction: Calculator\nAction Input: {"expression": "82000000 * 0.05"}', 35
                else:
                    return f"Thought: I have the revenue.\nFinal Answer: {obs}", 25

            if "multiply it by 10" in q_lower and "packet loss" in obs:
                if is_strong:
                    return 'Thought: Packet loss is 12%. Multiplying by 10.\nAction: Calculator\nAction Input: {"expression": "12 * 10"}', 35
                else:
                    return f"Thought: Packet loss reported.\nFinal Answer: {obs}", 25

            if "complex_task_data" in q_lower and "1200" in obs:
                if is_strong:
                    return 'Thought: Base value is 1200, multiplier 1.25, deduction 150. Calculating.\nAction: Calculator\nAction Input: {"expression": "(1200 * 1.25) - 150"}', 40
                else:
                    return f"Thought: Data retrieved.\nFinal Answer: 1200", 25

            if "profit margin (18%)" in q_lower and "45,000,000" in obs:
                if is_strong:
                    return 'Thought: Revenue is 45,000,000 and profit margin is 18%.\nAction: Calculator\nAction Input: {"expression": "45000000 * 0.18"}', 35
                else:
                    return f"Thought: Revenue retrieved.\nFinal Answer: {obs}", 25

            if "profit margin (22%)" in q_lower and "82,000,000" in obs:
                if is_strong:
                    return 'Thought: Revenue is 82,000,000 and profit margin is 22%.\nAction: Calculator\nAction Input: {"expression": "82000000 * 0.22"}', 35
                else:
                    return f"Thought: Revenue retrieved.\nFinal Answer: {obs}", 25

            if "difference between company_b revenue" in q_lower and ("82,000,000" in obs or "company_b" in obs):
                if is_strong:
                    return 'Thought: Company B is $82M and Company A is $45M. Calculating difference.\nAction: Calculator\nAction Input: {"expression": "82000000 - 45000000"}', 35
                else:
                    return f"Thought: Comparison done.\nFinal Answer: 82000000", 25

            if "50 units" in q_lower and "150" in obs:
                if is_strong:
                    # 150 * 0.90 * 50 = 6750
                    return 'Thought: Price is $150 with 10% discount ($135). Order of 50 units.\nAction: Calculator\nAction Input: {"expression": "150 * 0.90 * 50"}', 35
                else:
                    return f"Thought: Price retrieved.\nFinal Answer: 150", 25

            # Standard final answer from tool observation
            return f"Thought: The tool observation directly provides the required answer.\nFinal Answer: {obs}", 30

        # 2. Step 1: Initial tool selection or direct answer
        # Direct questions that don't need tools
        if "capital of france" in q_lower:
            return "Thought: I know this direct fact.\nFinal Answer: Paris", 15
        if "capital of japan" in q_lower:
            return "Thought: I know this direct fact.\nFinal Answer: Tokyo", 15

        # Code verification
        if "code" in q_lower or "syntax" in q_lower or "codeanalyzer" in q_lower or "python" in q_lower:
            code_match = re.search(r"'(.*?)'|\"(.*?)\"|```python(.*?)```|```(.*?)```", question, re.DOTALL)
            code_extracted = "def test(): return True"
            if code_match:
                for grp in code_match.groups():
                    if grp:
                        code_extracted = grp.strip()
                        break
            return f'Thought: I will use CodeAnalyzer to inspect the code snippet.\nAction: CodeAnalyzer\nAction Input: {{"code": "{code_extracted}"}}', 32

        # Knowledge lookup (checked before math so composite lookup+math queries look up data first)
        if "lookup" in q_lower or "revenue" in q_lower or "price" in q_lower or "ceo" in q_lower or "headquarters" in q_lower or "server_" in q_lower or "complex_task_data" in q_lower or "packet loss" in q_lower:
            query = "company_a"
            if "company_a" in q_lower: query = "revenue company_a" if "revenue" in q_lower or "profit" in q_lower else "ceo company_a"
            elif "company_b" in q_lower: query = "revenue company_b" if "revenue" in q_lower or "profit" in q_lower else "headquarters company_b"
            elif "server_alpha" in q_lower: query = "server_alpha status"
            elif "server_beta" in q_lower: query = "server_beta status"
            elif "product_widget_x" in q_lower or "widget x" in q_lower: query = "product_widget_x price"
            elif "complex_task_data" in q_lower: query = "complex_task_data"

            # If cheap model on hard composite task, it guesses incorrectly without tool
            is_hard_composite = "complex" in q_lower or "difference between" in q_lower or "50 units" in q_lower or "profit amount" in q_lower
            if is_hard_composite and not is_strong:
                return "Thought: I will guess directly.\nFinal Answer: 999999", 20

            return f'Thought: I will look up this information in the knowledge database first.\nAction: KnowledgeLookup\nAction Input: {{"query": "{query}"}}', 30

        # Hard reasoning tasks where cheap model struggles
        is_hard = "complex" in q_lower or "step-by-step" in q_lower or "profit amount" in q_lower or "difference between" in q_lower or "50 units" in q_lower or "annual return" in q_lower
        if is_hard and not is_strong:
            return "Thought: This looks like a complex calculation, I will guess directly.\nFinal Answer: 999999", 20

        # Math questions
        if "calculate" in q_lower or "sqrt" in q_lower or "+" in q_lower or "*" in q_lower or "/" in q_lower or "-" in q_lower:
            math_expr = re.search(r"calculate\s+([0-9\.\+\-\*\/\(\)\s\^\*sqrt]+)", question, re.IGNORECASE)
            expr = math_expr.group(1).strip() if math_expr else ""
            if not expr:
                m = re.search(r"([0-9\.\+\-\*\/\(\)\s]+)", question)
                expr = m.group(1).strip() if m else "10 + 10"
            expr = expr.replace("^", "**")
            return f'Thought: I will use the Calculator to compute the mathematical expression accurately.\nAction: Calculator\nAction Input: {{"expression": "{expr}"}}', 30

        return "Thought: I now have the final answer.\nFinal Answer: Complete", 20


class RealOpenAIProvider:
    """Provider communicating with any OpenAI-compatible API (Groq, OpenAI, etc.)."""
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        async with httpx.AsyncClient(timeout=settings.TIMEOUT_SECONDS) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 429:
                    raise RateLimitException(response.text)
                elif response.status_code >= 500:
                    raise LLMProviderException(f"Provider error: {response.text}", status_code=response.status_code, retryable=True)
                elif response.status_code != 200:
                    raise LLMProviderException(f"API Error ({response.status_code}): {response.text}", status_code=response.status_code, retryable=False)

                return response.json()
            except httpx.TimeoutException:
                raise TimeoutException(f"Request timed out after {settings.TIMEOUT_SECONDS}s")
            except httpx.RequestError as exc:
                raise LLMProviderException(f"Network error: {str(exc)}", status_code=502, retryable=True)


class ReliableLLMClient:
    """
    Reliability Layer:
    - Retries with exponential backoff on transient errors (429, timeouts, 5xx).
    - Automatic Fallback to secondary model if primary model fails all retries.
    """
    def __init__(self):
        self.mock_provider = MockLLMProvider()
        # Initialize real provider if key exists
        self.real_provider: Optional[RealOpenAIProvider] = None
        key = settings.GROQ_API_KEY or settings.OPENAI_API_KEY
        if key:
            self.real_provider = RealOpenAIProvider(api_key=key, base_url=settings.OPENAI_BASE_URL)

    def get_provider(self):
        """Returns real provider if configured, else mock provider."""
        return self.real_provider if self.real_provider else self.mock_provider

    async def execute_with_resilience(
        self,
        messages: List[Dict[str, str]],
        primary_model: str,
        fallback_model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes request with retries and fallback.
        Returns dict containing:
        - raw_response: response object
        - model_used: name of the model that successfully answered
        - fallback_used: boolean
        - latency_ms: duration in milliseconds
        """
        start_time = time.perf_counter()
        provider = self.get_provider()
        fallback_target = fallback_model or settings.FALLBACK_MODEL

        # Attempt primary model with retries
        try:
            resp = await self._retry_loop(provider, messages, primary_model, temperature, **kwargs)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "raw_response": resp,
                "model_used": primary_model,
                "fallback_used": False,
                "latency_ms": latency_ms
            }
        except Exception as primary_error:
            # If fallback is the same as primary, re-raise
            if fallback_target == primary_model:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                raise primary_error

            # Fallback to secondary model
            try:
                resp = await self._retry_loop(provider, messages, fallback_target, temperature, **kwargs)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return {
                    "raw_response": resp,
                    "model_used": fallback_target,
                    "fallback_used": True,
                    "latency_ms": latency_ms,
                    "fallback_reason": str(primary_error)
                }
            except Exception as fallback_error:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                raise LLMProviderException(
                    f"Both primary model '{primary_model}' and fallback model '{fallback_target}' failed. Last error: {str(fallback_error)}"
                )

    async def _retry_loop(self, provider, messages, model, temperature, **kwargs):
        last_exception = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                return await provider.generate(messages, model=model, temperature=temperature, **kwargs)
            except LLMProviderException as exc:
                last_exception = exc
                if not exc.retryable or attempt == settings.MAX_RETRIES:
                    raise exc
                # Exponential backoff with jitter
                delay = (settings.RETRY_BACKOFF_FACTOR ** attempt) * 0.1 + random.uniform(0.01, 0.05)
                await asyncio.sleep(delay)
            except Exception as exc:
                last_exception = exc
                if attempt == settings.MAX_RETRIES:
                    raise exc
                delay = (settings.RETRY_BACKOFF_FACTOR ** attempt) * 0.1
                await asyncio.sleep(delay)
        raise last_exception or LLMProviderException("Retry loop exhausted")


reliable_client = ReliableLLMClient()
