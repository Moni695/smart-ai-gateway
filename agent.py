"""
Tool-Using Agent with 3 tools (Calculator, KnowledgeLookup, CodeAnalyzer).
Implements a ReAct loop and calls LLMs exclusively through the AI Gateway.
Records trajectory: steps, tool invocations, and gateway telemetry.
"""

import math
import json
import re
from typing import Dict, Any, List, Optional
from config import RoutePolicy


class CalculatorTool:
    """Safe mathematical evaluator supporting arithmetic and basic math functions."""
    name = "Calculator"
    description = "Useful for evaluating mathematical and arithmetic expressions."

    def execute(self, expression: str) -> str:
        try:
            # Allow safe characters: alphanumeric, operators, parentheses
            cleaned = re.sub(r"[^a-zA-Z0-9\+\-\*\/\(\)\.\,\s_]", "", expression)
            # Safe eval namespace
            safe_dict = {
                "sqrt": math.sqrt, "pow": math.pow, "abs": abs,
                "round": round, "sin": math.sin, "cos": math.cos,
                "pi": math.pi, "e": math.e
            }
            res = eval(cleaned, {"__builtins__": None}, safe_dict)
            return str(round(res, 4) if isinstance(res, float) else res)
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"


class KnowledgeLookupTool:
    """Mock fact & enterprise database lookup."""
    name = "KnowledgeLookup"
    description = "Useful for looking up company financials, server statuses, and verified facts."

    DATABASE = {
        "revenue company_a": "Company_A 2025 revenue was $45,000,000 with a profit margin of 18%.",
        "revenue company_b": "Company_B 2025 revenue was $82,000,000 with a profit margin of 22%.",
        "capital france": "Paris",
        "capital japan": "Tokyo",
        "server_alpha status": "Server Alpha is ONLINE with 99.98% uptime in us-east-1.",
        "server_beta status": "Server Beta is DEGRADED with 12% packet loss in eu-west-1.",
        "product_widget_x price": "Widget X is priced at $150 per unit with 10% volume discount.",
        "ceo company_a": "Alice Zhang",
        "headquarters company_b": "Berlin, Germany",
        "complex_task_data": "Entity X base value is 1200, growth multiplier is 1.25, and tax deduction is 150."
    }

    def execute(self, query: str) -> str:
        query_norm = query.lower().strip()
        # Direct match or fuzzy keyword search
        for key, val in self.DATABASE.items():
            if key in query_norm or all(w in query_norm for w in key.split()):
                return val
        # Substring search
        for key, val in self.DATABASE.items():
            words = [w for w in query_norm.split() if len(w) > 3]
            if words and any(w in key for w in words):
                return val
        return f"No record found for query: '{query}'"


class CodeAnalyzerTool:
    """Analyzes Python code snippets for syntax errors and line count."""
    name = "CodeAnalyzer"
    description = "Useful for inspecting and validating Python code syntax."

    def execute(self, code: str) -> str:
        code_str = code.strip()
        if code_str.startswith("```"):
            lines = code_str.splitlines()
            code_str = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        try:
            compile(code_str, "<string>", "exec")
            return f"Valid syntax! Parsed {len(code_str.splitlines())} lines successfully."
        except SyntaxError as e:
            return f"SyntaxError at line {e.lineno}: {e.msg}"


class ToolAgent:
    """
    ReAct Agent capable of autonomous multi-step reasoning and tool execution.
    Sends LLM queries through the AI Gateway under a specified policy.
    """
    def __init__(self, gateway_completion_fn=None):
        self.tools = {
            CalculatorTool.name: CalculatorTool(),
            KnowledgeLookupTool.name: KnowledgeLookupTool(),
            CodeAnalyzerTool.name: CodeAnalyzerTool()
        }
        self.gateway_completion_fn = gateway_completion_fn

    def get_tool_descriptions(self) -> str:
        return "\n".join([f"- {name}: {t.description}" for name, t in self.tools.items()])

    async def run(
        self,
        task: str,
        policy: RoutePolicy = RoutePolicy.AUTO_ROUTER,
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Runs the ReAct agent on a task.
        Returns full trajectory, tools invoked, final answer, and telemetry.
        """
        system_prompt = (
            "You are an AI assistant equipped with tools to answer questions.\n"
            "You have access to the following tools:\n"
            f"{self.get_tool_descriptions()}\n\n"
            "Use the following format:\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [Calculator, KnowledgeLookup, CodeAnalyzer]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {task}"}
        ]

        steps: List[Dict[str, Any]] = []
        tools_used: List[str] = []
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0

        for step_idx in range(1, max_steps + 1):
            # Call Gateway
            gw_result = await self.gateway_completion_fn(messages, policy=policy)
            content = gw_result["content"]
            model_used = gw_result.get("model_used", "unknown")
            tokens = gw_result.get("tokens", 0)
            cost = gw_result.get("cost", 0.0)
            lat = gw_result.get("latency_ms", 0.0)

            total_tokens += tokens
            total_cost += cost
            total_latency += lat

            # Check for Final Answer
            if "Final Answer:" in content:
                final_answer = content.split("Final Answer:")[-1].strip()
                steps.append({
                    "step": step_idx,
                    "model": model_used,
                    "content": content,
                    "type": "final_answer"
                })
                return {
                    "task": task,
                    "final_answer": final_answer,
                    "tools_used": list(set(tools_used)),
                    "steps": steps,
                    "step_count": step_idx,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "total_latency_ms": round(total_latency, 2),
                    "status": "completed"
                }

            # Parse Action and Action Input
            action_match = re.search(r"Action:\s*([A-Za-z0-9_]+)", content)
            input_match = re.search(r"Action Input:\s*(.+)", content, re.DOTALL)

            if action_match:
                tool_name = action_match.group(1).strip()
                raw_input = input_match.group(1).strip() if input_match else ""
                
                # Try parsing JSON input if formatted as JSON
                tool_arg = raw_input
                try:
                    if raw_input.startswith("{"):
                        parsed_json = json.loads(raw_input)
                        tool_arg = parsed_json.get("expression") or parsed_json.get("query") or parsed_json.get("code") or list(parsed_json.values())[0]
                except Exception:
                    pass

                tools_used.append(tool_name)
                tool = self.tools.get(tool_name)
                obs = tool.execute(str(tool_arg)) if tool else f"Error: Tool '{tool_name}' not recognized."

                steps.append({
                    "step": step_idx,
                    "model": model_used,
                    "content": content,
                    "tool": tool_name,
                    "tool_input": tool_arg,
                    "observation": obs
                })

                # Append to context
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                # No clear action or final answer, end gracefully
                steps.append({"step": step_idx, "model": model_used, "content": content, "type": "unstructured"})
                return {
                    "task": task,
                    "final_answer": content,
                    "tools_used": list(set(tools_used)),
                    "steps": steps,
                    "step_count": step_idx,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                    "total_latency_ms": round(total_latency, 2),
                    "status": "partial"
                }

        return {
            "task": task,
            "final_answer": "Max steps reached without definitive answer",
            "tools_used": list(set(tools_used)),
            "steps": steps,
            "step_count": max_steps,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "total_latency_ms": round(total_latency, 2),
            "status": "exhausted"
        }
