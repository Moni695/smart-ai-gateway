"""
Scorer module for Agent Evaluations.
Computes multi-dimensional scores across Answer Correctness, Tool Selection,
and Trajectory Efficiency.
"""

import re
from typing import Dict, Any, List


class EvalScorer:
    """Multi-dimensional agent evaluator."""

    @staticmethod
    def normalize_str(s: str) -> str:
        if not s:
            return ""
        # Remove currency, commas, extra whitespace
        cleaned = re.sub(r"[\$\,\s]", "", s).lower()
        return cleaned

    def score_answer(self, final_answer: str, expected_answer: str) -> float:
        if not final_answer or not expected_answer:
            return 0.0

        norm_final = self.normalize_str(final_answer)
        norm_expected = self.normalize_str(expected_answer)

        # 1. Exact or substring match
        if norm_expected in norm_final:
            return 1.0

        # 2. Numerical extraction match
        num_expected = re.findall(r"[-+]?\d*\.?\d+", norm_expected)
        num_final = re.findall(r"[-+]?\d*\.?\d+", norm_final)

        if num_expected and num_final:
            try:
                exp_val = float(num_expected[0])
                for fin_val_str in num_final:
                    fin_val = float(fin_val_str)
                    # 1% tolerance
                    if abs(fin_val - exp_val) <= max(0.01, 0.01 * abs(exp_val)):
                        return 1.0
            except ValueError:
                pass

        # 3. Partial keyword overlap
        words_expected = set(expected_answer.lower().split())
        words_final = set(final_answer.lower().split())
        if words_expected and words_expected.issubset(words_final):
            return 0.85

        return 0.0

    def score_tools(self, tools_used: List[str], expected_tools: List[str]) -> float:
        set_used = set(tools_used or [])
        set_exp = set(expected_tools or [])

        if not set_exp:
            # If no tools were expected, reward zero tools used
            return 1.0 if not set_used else 0.70

        if not set_used:
            return 0.0

        # Jaccard / Overlap score
        intersection = len(set_used.intersection(set_exp))
        recall = intersection / len(set_exp)
        precision = intersection / len(set_used)

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(f1, 2)

    def score_trajectory(self, step_count: int, max_expected_steps: int) -> float:
        if step_count <= max_expected_steps:
            return 1.0
        # Gentle penalty for taking extra steps
        excess = step_count - max_expected_steps
        score = max(0.0, 1.0 - (excess * 0.25))
        return round(score, 2)

    def evaluate_task(
        self,
        final_answer: str,
        tools_used: List[str],
        step_count: int,
        task_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates composite evaluation for a completed agent task.
        """
        answer_score = self.score_answer(final_answer, str(task_meta["expected_answer"]))
        tool_score = self.score_tools(tools_used, task_meta.get("expected_tools", []))
        traj_score = self.score_trajectory(step_count, task_meta.get("max_expected_steps", 3))

        composite_score = round(
            (0.50 * answer_score) + (0.30 * tool_score) + (0.20 * traj_score),
            3
        )
        passed = composite_score >= 0.70

        return {
            "task_id": task_meta["id"],
            "difficulty": task_meta["difficulty"],
            "category": task_meta.get("category", "general"),
            "answer_score": answer_score,
            "tool_score": tool_score,
            "trajectory_score": traj_score,
            "composite_score": composite_score,
            "passed": passed
        }


eval_scorer = EvalScorer()
