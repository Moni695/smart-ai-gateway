"""
Auto-Routing Engine for the AI Gateway.
Evaluates incoming prompt complexity using explainable heuristics (length, keywords, code presence,
constraint density) and selects the optimal model tier (cheap vs strong).
"""

import re
from typing import List, Dict, Any, Tuple
from config import settings, RoutePolicy


class ComplexityRouter:
    """
    Explainable complexity evaluator and model router.
    Routes easy tasks to cheap, fast models and hard tasks to strong models.
    """
    REASONING_KEYWORDS = [
        "step-by-step", "prove", "proof", "derivation", "derive", "optimize", "optimization",
        "refactor", "algorithmic", "time complexity", "space complexity", "architecture",
        "edge cases", "formal logic", "trade-offs", "multi-step", "constraint", "theorem",
        "complex", "profit", "difference between", "compound", "annual", "syntax error",
        "packet loss", "order of"
    ]

    CODE_PATTERNS = [
        r"```[\s\S]*?```",
        r"\bdef\s+[a-zA-Z_]\w*\s*\(",
        r"\bclass\s+[a-zA-Z_]\w*[:\(]",
        r"\bSELECT\s+.+\s+FROM\b",
        r"\bimport\s+[a-zA-Z_]\w*",
        r"\bconst\s+[a-zA-Z_]\w*\s*=",
        r"\{[\s\S]*?:[\s\S]*?\}"
    ]

    SIMPLE_KEYWORDS = [
        "hello", "hi", "hey", "good morning", "thank you", "thanks",
        "who is", "what is the capital", "translate to", "summarize in 1 sentence"
    ]

    def __init__(self, threshold: float = None):
        self.threshold = threshold if threshold is not None else settings.COMPLEXITY_THRESHOLD

    def evaluate_complexity(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Tuple[float, str]:
        """
        Calculates complexity score (0.0 to 1.0) and generates an explanatory reason.
        """
        # Extract full prompt text
        full_text = " ".join([
            str(m.get("content", "")) for m in messages if isinstance(m.get("content"), (str, int, float))
        ])
        lower_text = full_text.lower()
        reasons: List[str] = []
        score = 0.0

        # 1. Simple heuristic quick-check (negative weight)
        is_short = len(full_text.split()) < 15
        if is_short and any(s in lower_text for s in self.SIMPLE_KEYWORDS):
            return 0.1, "Simple greeting or trivial direct lookup detected"

        # 2. Length / Token density (Prompts with rich context or long instructions)
        word_count = len(full_text.split())
        if word_count > 300:
            score += 0.35
            reasons.append(f"High token length ({word_count} words)")
        elif word_count > 120:
            score += 0.20
            reasons.append(f"Moderate token length ({word_count} words)")

        # 3. Code Presence
        code_matches = sum(1 for pat in self.CODE_PATTERNS if re.search(pat, full_text, re.IGNORECASE))
        if code_matches > 0:
            code_weight = min(0.40, code_matches * 0.20)
            score += code_weight
            reasons.append(f"Code syntax/blocks detected ({code_matches} markers)")

        # 4. Complex Reasoning / Logic Keywords
        matched_keywords = [kw for kw in self.REASONING_KEYWORDS if kw in lower_text]
        if matched_keywords:
            kw_weight = min(0.35, len(matched_keywords) * 0.15)
            score += kw_weight
            reasons.append(f"Reasoning keywords detected: {', '.join(matched_keywords[:3])}")

        # 5. Tools presence / Multi-agent schema
        if tools and len(tools) > 0:
            score += 0.25
            reasons.append(f"Agent tool schema present ({len(tools)} tools)")

        # Clamp between 0.0 and 1.0
        final_score = round(max(0.0, min(1.0, score)), 3)
        summary_reason = "; ".join(reasons) if reasons else "Standard direct prompt without high complexity markers"

        return final_score, summary_reason

    def route(
        self,
        messages: List[Dict[str, Any]],
        policy: RoutePolicy = RoutePolicy.AUTO_ROUTER,
        requested_model: str = None,
        tools: List[Dict[str, Any]] = None
    ) -> Tuple[str, str, float]:
        """
        Selects model based on policy and complexity evaluation.
        Returns: (selected_model, routing_reason, complexity_score)
        """
        score, reason = self.evaluate_complexity(messages, tools)

        if policy == RoutePolicy.ALWAYS_CHEAP:
            return settings.CHEAP_MODEL, f"Policy override: ALWAYS_CHEAP (eval score: {score})", score

        if policy == RoutePolicy.ALWAYS_STRONG:
            return settings.STRONG_MODEL, f"Policy override: ALWAYS_STRONG (eval score: {score})", score

        # AUTO_ROUTER decision
        if score >= self.threshold:
            chosen = settings.STRONG_MODEL
            full_reason = f"Routed to Strong Model ({chosen}) - Score {score} >= Threshold {self.threshold}. Signals: {reason}"
        else:
            chosen = settings.CHEAP_MODEL
            full_reason = f"Routed to Cheap Model ({chosen}) - Score {score} < Threshold {self.threshold}. Signals: {reason}"

        return chosen, full_reason, score


router = ComplexityRouter()
