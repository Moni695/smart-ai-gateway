"""
Evaluation Dataset for the AI Gateway.
Contains 25 fixed tasks across Easy, Medium, and Hard tiers with verified ground truth answers,
expected tools, and difficulty levels.
"""

from typing import List, Dict, Any

EVAL_DATASET: List[Dict[str, Any]] = [
    # --- EASY TASKS (Single-step or direct answer) ---
    {
        "id": "easy-1",
        "category": "direct_qa",
        "difficulty": "easy",
        "task": "What is the capital of France?",
        "expected_answer": "Paris",
        "expected_tools": [],
        "max_expected_steps": 1
    },
    {
        "id": "easy-2",
        "category": "direct_qa",
        "difficulty": "easy",
        "task": "What is the capital of Japan?",
        "expected_answer": "Tokyo",
        "expected_tools": [],
        "max_expected_steps": 1
    },
    {
        "id": "easy-3",
        "category": "math",
        "difficulty": "easy",
        "task": "Calculate 45 + 55",
        "expected_answer": "100",
        "expected_tools": ["Calculator"],
        "max_expected_steps": 2
    },
    {
        "id": "easy-4",
        "category": "math",
        "difficulty": "easy",
        "task": "Calculate 12 * 8",
        "expected_answer": "96",
        "expected_tools": ["Calculator"],
        "max_expected_steps": 2
    },
    {
        "id": "easy-5",
        "category": "lookup",
        "difficulty": "easy",
        "task": "What is the price of product_widget_x?",
        "expected_answer": "150",
        "expected_tools": ["KnowledgeLookup"],
        "max_expected_steps": 2
    },
    {
        "id": "easy-6",
        "category": "lookup",
        "difficulty": "easy",
        "task": "Who is the CEO of Company_A?",
        "expected_answer": "Alice Zhang",
        "expected_tools": ["KnowledgeLookup"],
        "max_expected_steps": 2
    },
    {
        "id": "easy-7",
        "category": "lookup",
        "difficulty": "easy",
        "task": "Where is the headquarters of Company_B located?",
        "expected_answer": "Berlin",
        "expected_tools": ["KnowledgeLookup"],
        "max_expected_steps": 2
    },
    {
        "id": "easy-8",
        "category": "math",
        "difficulty": "easy",
        "task": "Calculate 500 / 25",
        "expected_answer": "20",
        "expected_tools": ["Calculator"],
        "max_expected_steps": 2
    },

    # --- MEDIUM TASKS (Multi-step tool combination or lookup + math) ---
    {
        "id": "med-1",
        "category": "composite",
        "difficulty": "medium",
        "task": "Lookup the revenue of Company_A and calculate 10 percent of it.",
        "expected_answer": "4500000",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 3
    },
    {
        "id": "med-2",
        "category": "composite",
        "difficulty": "medium",
        "task": "Lookup the revenue of Company_B and calculate 5 percent of it.",
        "expected_answer": "4100000",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 3
    },
    {
        "id": "med-3",
        "category": "math",
        "difficulty": "medium",
        "task": "Calculate (15 * 40) + (250 / 5)",
        "expected_answer": "650",
        "expected_tools": ["Calculator"],
        "max_expected_steps": 2
    },
    {
        "id": "med-4",
        "category": "code",
        "difficulty": "medium",
        "task": "Use CodeAnalyzer to check if the following python code is valid: 'def test(): return True'",
        "expected_answer": "valid",
        "expected_tools": ["CodeAnalyzer"],
        "max_expected_steps": 2
    },
    {
        "id": "med-5",
        "category": "lookup",
        "difficulty": "medium",
        "task": "Check the status of server_alpha and report its uptime percentage.",
        "expected_answer": "99.98",
        "expected_tools": ["KnowledgeLookup"],
        "max_expected_steps": 2
    },
    {
        "id": "med-6",
        "category": "composite",
        "difficulty": "medium",
        "task": "Find the packet loss on server_beta and multiply it by 10.",
        "expected_answer": "120",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 3
    },
    {
        "id": "med-7",
        "category": "math",
        "difficulty": "medium",
        "task": "Calculate the square root of 625 using calculator: sqrt(625)",
        "expected_answer": "25",
        "expected_tools": ["Calculator"],
        "max_expected_steps": 2
    },
    {
        "id": "med-8",
        "category": "code",
        "difficulty": "medium",
        "task": "Check syntax of python code: 'for x in range(10): print(x * 2)'",
        "expected_answer": "valid",
        "expected_tools": ["CodeAnalyzer"],
        "max_expected_steps": 2
    },

    # --- HARD TASKS (Complex multi-constraint, algorithmic reasoning, code syntax edge cases) ---
    {
        "id": "hard-1",
        "category": "reasoning_composite",
        "difficulty": "hard",
        "task": "Look up complex_task_data, take base value, multiply by growth multiplier, and subtract tax deduction.",
        "expected_answer": "1350",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 4
    },
    {
        "id": "hard-2",
        "category": "code_logic",
        "difficulty": "hard",
        "task": "Analyze the following Python snippet and check for syntax errors: 'def broken(x) return x + 1'",
        "expected_answer": "SyntaxError",
        "expected_tools": ["CodeAnalyzer"],
        "max_expected_steps": 2
    },
    {
        "id": "hard-3",
        "category": "multi_step_math",
        "difficulty": "hard",
        "task": "Compute step-by-step optimization: calculate (350 * 12) + (1400 / 7) - 150",
        "expected_answer": "4250",
        "expected_tools": ["Calculator"],
        "max_expected_steps": 3
    },
    {
        "id": "hard-4",
        "category": "reasoning",
        "difficulty": "hard",
        "task": "Find the profit amount in dollars of Company_A: calculate revenue ($45M) multiplied by profit margin (18%).",
        "expected_answer": "8100000",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 3
    },
    {
        "id": "hard-5",
        "category": "reasoning",
        "difficulty": "hard",
        "task": "Find the profit amount in dollars of Company_B: calculate revenue ($82M) multiplied by profit margin (22%).",
        "expected_answer": "18040000",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 3
    },
    {
        "id": "hard-6",
        "category": "multi_step_math",
        "difficulty": "hard",
        "task": "Calculate compound annual return for initial 1000 at 10% across 3 years: 1000 * (1.1 ** 3)",
        "expected_answer": "1331",
        "expected_tools": ["Calculator"],
        "max_expected_steps": 2
    },
    {
        "id": "hard-7",
        "category": "code_logic",
        "difficulty": "hard",
        "task": "Validate code containing list comprehension: 'result = [x**2 for x in range(10) if x % 2 == 0]'",
        "expected_answer": "valid",
        "expected_tools": ["CodeAnalyzer"],
        "max_expected_steps": 2
    },
    {
        "id": "hard-8",
        "category": "multi_constraint",
        "difficulty": "hard",
        "task": "If Widget X price is $150 with 10% discount, calculate the total price for an order of 50 units.",
        "expected_answer": "6750",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 3
    },
    {
        "id": "hard-9",
        "category": "reasoning_composite",
        "difficulty": "hard",
        "task": "Calculate difference between Company_B revenue ($82M) and Company_A revenue ($45M).",
        "expected_answer": "37000000",
        "expected_tools": ["KnowledgeLookup", "Calculator"],
        "max_expected_steps": 3
    }
]
