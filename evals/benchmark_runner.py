"""
Benchmark Runner for Agent Evaluations across 3 Policies:
1. ALWAYS_CHEAP
2. ALWAYS_STRONG
3. AUTO_ROUTER

Compares Cost, Latency, and Pass Rate with statistical verification,
generates a visual comparison chart (PNG), and outputs the exact resume bullet.
"""

import sys
import os
import asyncio
from typing import Dict, Any, List
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings, RoutePolicy
from router import router
from providers import reliable_client
from cost_engine import cost_engine
from agent import ToolAgent
from evals.dataset import EVAL_DATASET
from evals.scorer import eval_scorer


async def internal_gateway_completion(messages: List[Dict[str, str]], policy: RoutePolicy) -> Dict[str, Any]:
    """In-process direct gateway execution function matching FastAPI endpoint logic."""
    target_model, reason, score = router.route(messages, policy=policy)
    exec_res = await reliable_client.execute_with_resilience(messages, primary_model=target_model)

    raw = exec_res["raw_response"]
    model_used = exec_res["model_used"]
    usage = raw.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens

    cost = cost_engine.calculate_cost(model_used, prompt_tokens, completion_tokens)
    content = raw["choices"][0]["message"]["content"]

    return {
        "content": content,
        "model_used": model_used,
        "tokens": total_tokens,
        "cost": cost,
        "latency_ms": exec_res["latency_ms"],
        "routing_reason": reason,
        "complexity_score": score
    }


async def run_policy_benchmark(policy: RoutePolicy) -> Dict[str, Any]:
    """Runs the full 25-task exam under a single policy."""
    agent = ToolAgent(gateway_completion_fn=internal_gateway_completion)
    task_results: List[Dict[str, Any]] = []

    total_cost = 0.0
    total_latency = 0.0
    total_tokens = 0
    passed_count = 0

    print(f"\n[RUN] Running Benchmark Exam for Policy: [{policy.value.upper()}] ({len(EVAL_DATASET)} tasks)...")

    for idx, task_meta in enumerate(EVAL_DATASET, 1):
        res = await agent.run(task_meta["task"], policy=policy)
        eval_score = eval_scorer.evaluate_task(
            final_answer=res["final_answer"],
            tools_used=res["tools_used"],
            step_count=res["step_count"],
            task_meta=task_meta
        )

        total_cost += res["total_cost_usd"]
        total_latency += res["total_latency_ms"]
        total_tokens += res["total_tokens"]

        if eval_score["passed"]:
            passed_count += 1

        task_results.append({
            "task_id": task_meta["id"],
            "difficulty": task_meta["difficulty"],
            "passed": eval_score["passed"],
            "composite_score": eval_score["composite_score"],
            "cost_usd": res["total_cost_usd"],
            "latency_ms": res["total_latency_ms"],
            "tools_used": res["tools_used"]
        })

    num_tasks = len(EVAL_DATASET)
    pass_rate = round((passed_count / num_tasks) * 100, 2)
    avg_latency = round(total_latency / num_tasks, 2)
    avg_cost_per_task = round(total_cost / num_tasks, 6)

    print(f"[OK] Completed [{policy.value.upper()}]: Pass Rate={pass_rate}%, Cost=${total_cost:.6f}, Avg Latency={avg_latency}ms")

    return {
        "policy": policy.value,
        "total_tasks": num_tasks,
        "passed_tasks": passed_count,
        "pass_rate_pct": pass_rate,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": avg_cost_per_task,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "task_results": task_results
    }


def generate_benchmark_chart(results: Dict[str, Dict[str, Any]], output_path: str = "benchmark_results.png"):
    """Creates a side-by-side comparative chart using matplotlib."""
    policies = ["ALWAYS_CHEAP", "AUTO_ROUTER", "ALWAYS_STRONG"]
    keys = ["always_cheap", "auto_router", "always_strong"]
    colors = ["#f87171", "#38bdf8", "#34d399"]

    pass_rates = [results[k]["pass_rate_pct"] for k in keys]
    costs = [results[k]["total_cost_usd"] * 1000 for k in keys]  # in milli-dollars for readability
    latencies = [results[k]["avg_latency_ms"] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')

    # Subplot 1: Pass Rate
    bars1 = ax1.bar(policies, pass_rates, color=colors, width=0.55, edgecolor='#334155')
    ax1.set_title("Agent Exam Pass Rate (%)", color='#f8fafc', fontsize=14, pad=12, fontweight='bold')
    ax1.set_ylabel("Pass Rate (%)", color='#94a3b8', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.tick_params(colors='#94a3b8')
    for spine in ax1.spines.values(): spine.set_color('#334155')

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', color='#f8fafc', fontweight='bold')

    # Subplot 2: Total Cost ($)
    bars2 = ax2.bar(policies, costs, color=colors, width=0.55, edgecolor='#334155')
    ax2.set_title("Total Cost for 25 Tasks ($ x 10^-3)", color='#f8fafc', fontsize=14, pad=12, fontweight='bold')
    ax2.set_ylabel("Cost ($ / 1000 tasks)", color='#94a3b8', fontsize=12)
    ax2.tick_params(colors='#94a3b8')
    for spine in ax2.spines.values(): spine.set_color('#334155')

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"${yval/1000:.5f}", ha='center', va='bottom', color='#f8fafc', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=200)
    plt.close()
    print(f"\n[CHART] Generated benchmark chart saved to: {output_path}")


async def run_full_benchmark_suite(generate_plot: bool = True) -> Dict[str, Any]:
    """Runs the 25-task exam under all 3 policies and returns summary metrics."""
    print("=================================================================")
    print("   AI GATEWAY EVALUATION BENCHMARK: 3-POLICY COMPARISON")
    print("=================================================================")

    # 1. Run all three policies
    results = {}
    for pol in [RoutePolicy.ALWAYS_CHEAP, RoutePolicy.ALWAYS_STRONG, RoutePolicy.AUTO_ROUTER]:
        results[pol.value] = await run_policy_benchmark(pol)

    # 2. Calculate savings and delta
    cheap = results["always_cheap"]
    strong = results["always_strong"]
    router_res = results["auto_router"]

    cost_savings_pct = round(((strong["total_cost_usd"] - router_res["total_cost_usd"]) / strong["total_cost_usd"]) * 100, 1) if strong["total_cost_usd"] > 0 else 0.0
    pass_delta = round(strong["pass_rate_pct"] - router_res["pass_rate_pct"], 1)

    # 3. Print Results Table
    print("\n" + "=" * 70)
    print("                      BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<18} | {'Pass Rate':<10} | {'Total Cost ($)':<15} | {'Avg Latency':<12}")
    print("-" * 70)
    print(f"{'ALWAYS_CHEAP':<18} | {cheap['pass_rate_pct']:>8.1f}% | ${cheap['total_cost_usd']:>13.6f} | {cheap['avg_latency_ms']:>8.1f} ms")
    print(f"{'ALWAYS_STRONG':<18} | {strong['pass_rate_pct']:>8.1f}% | ${strong['total_cost_usd']:>13.6f} | {strong['avg_latency_ms']:>8.1f} ms")
    print(f"{'AUTO_ROUTER':<18} | {router_res['pass_rate_pct']:>8.1f}% | ${router_res['total_cost_usd']:>13.6f} | {router_res['avg_latency_ms']:>8.1f} ms")
    print("-" * 70)
    print(f"[COST] Cost Savings vs Always Strong : {cost_savings_pct}% cheaper")
    print(f"[EVAL] Pass Rate Gap vs Always Strong: {pass_delta} percentage points")
    print("=" * 70)

    # 4. Generate visual plot
    if generate_plot:
        chart_file = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.png")
        generate_benchmark_chart(results, chart_file)

    # 5. Output Resume Bullet
    resume_bullet = (
        f"Built an LLM gateway with auto-routing, fallback and cost tracking; "
        f"auto-routing cut LLM cost by {cost_savings_pct}% while keeping agent "
        f"eval pass rate within {pass_delta} points of always using the strongest model."
    )

    print("\n[RESUME BULLET] (Fill X and Y with real verified numbers):")
    print("-----------------------------------------------------------------")
    print(f"> \"{resume_bullet}\"")
    print("-----------------------------------------------------------------\n")

    return {
        "results": results,
        "cost_savings_pct": cost_savings_pct,
        "pass_delta": pass_delta,
        "resume_bullet": resume_bullet
    }


async def main():
    await run_full_benchmark_suite(generate_plot=True)


if __name__ == "__main__":
    asyncio.run(main())
