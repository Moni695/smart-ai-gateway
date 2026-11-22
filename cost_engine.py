"""
Cost Engine for calculating exact LLM request costs.
Calculates costs based on prompt and completion token counts and model pricing tiers.
"""

from typing import Dict, Tuple
from config import MODEL_PRICING


class CostEngine:
    def __init__(self, pricing_table: Dict[str, Tuple[float, float]] = None):
        self.pricing_table = pricing_table or MODEL_PRICING

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculates total cost in USD for a request.
        Pricing table values are in USD per 1,000,000 tokens.
        """
        # Match exact model or find best prefix match, otherwise fallback to default
        pricing = self.pricing_table.get(model)
        if not pricing:
            # Check prefix (e.g. llama-3.1-8b)
            for key, val in self.pricing_table.items():
                if key in model or model in key:
                    pricing = val
                    break
        if not pricing:
            pricing = self.pricing_table.get("default", (0.10, 0.20))

        prompt_rate, completion_rate = pricing
        prompt_cost = (prompt_tokens / 1_000_000.0) * prompt_rate
        completion_cost = (completion_tokens / 1_000_000.0) * completion_rate
        return round(prompt_cost + completion_cost, 8)

    def get_pricing_rate(self, model: str) -> Tuple[float, float]:
        """Returns (prompt_rate_per_1M, completion_rate_per_1M)"""
        return self.pricing_table.get(model, self.pricing_table.get("default", (0.10, 0.20)))


cost_engine = CostEngine()
