"""
Unit and Integration Test Suite for the Smart AI Gateway.
Covers routing, cost calculation, response caching, fallback/retries,
agent tools, eval scorers, and FastAPI endpoints.
"""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from config import settings, RoutePolicy
from cost_engine import CostEngine
from router import ComplexityRouter
from cache import TTLCache
from providers import ReliableLLMClient, MockLLMProvider
from agent import CalculatorTool, KnowledgeLookupTool, CodeAnalyzerTool
from evals.scorer import EvalScorer
from storage import DatabaseManager
from main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# 1. Cost Engine Tests
def test_cost_calculation():
    engine = CostEngine()
    # LLaMA-3.1-8b is $0.05 / 1M prompt, $0.08 / 1M completion
    cost = engine.calculate_cost("llama-3.1-8b-instant", 1000, 500)
    expected = (1000 / 1_000_000 * 0.05) + (500 / 1_000_000 * 0.08)
    assert abs(cost - expected) < 1e-7
    assert cost > 0


# 2. Router Tests
def test_router_heuristics():
    router = ComplexityRouter(threshold=0.45)

    # Simple prompt -> cheap model
    simple_msgs = [{"role": "user", "content": "Hello, how are you today?"}]
    model, reason, score = router.route(simple_msgs, policy=RoutePolicy.AUTO_ROUTER)
    assert model == settings.CHEAP_MODEL
    assert score < 0.45

    # Complex prompt with code & reasoning -> strong model
    complex_msgs = [{
        "role": "user",
        "content": "Can you refactor this python function step-by-step and optimize time complexity:\n```python\ndef solve(arr):\n    return sorted(arr)\n```"
    }]
    model, reason, score = router.route(complex_msgs, policy=RoutePolicy.AUTO_ROUTER)
    assert model == settings.STRONG_MODEL
    assert score >= 0.45

    # Policy overrides
    forced_cheap, _, _ = router.route(complex_msgs, policy=RoutePolicy.ALWAYS_CHEAP)
    assert forced_cheap == settings.CHEAP_MODEL

    forced_strong, _, _ = router.route(simple_msgs, policy=RoutePolicy.ALWAYS_STRONG)
    assert forced_strong == settings.STRONG_MODEL


# 3. Cache Tests
def test_ttl_cache():
    cache = TTLCache(default_ttl=60)
    messages = [{"role": "user", "content": "What is 2+2?"}]

    # Initially empty
    assert cache.get(messages, model="test-model") is None

    # Set cache
    response_payload = {"choices": [{"message": {"content": "4"}}], "model": "test-model"}
    cache.set(messages, model="test-model", response=response_payload)

    # Cache hit
    cached = cache.get(messages, model="test-model")
    assert cached is not None
    assert cached["cache_hit"] is True
    assert cached["choices"][0]["message"]["content"] == "4"


# 4. Resilience & Fallback Tests
@pytest.mark.asyncio
async def test_fallback_on_provider_error():
    client = ReliableLLMClient()
    mock_prov: MockLLMProvider = client.mock_provider

    # Inject rate limit error into primary model
    primary = settings.STRONG_MODEL
    fallback = settings.CHEAP_MODEL
    mock_prov.set_simulation_error(primary, "rate_limit")

    try:
        messages = [{"role": "user", "content": "test message"}]
        result = await client.execute_with_resilience(
            messages=messages,
            primary_model=primary,
            fallback_model=fallback
        )
        assert result["fallback_used"] is True
        assert result["model_used"] == fallback
    finally:
        # Clean up simulation
        mock_prov.set_simulation_error(primary, None)


# 5. Agent Tools Tests
def test_agent_tools():
    calc = CalculatorTool()
    assert calc.execute("40 + 2") == "42"
    assert calc.execute("sqrt(16)") == "4.0"

    lookup = KnowledgeLookupTool()
    assert "Paris" in lookup.execute("capital france")
    assert "45,000,000" in lookup.execute("revenue company_a")

    code_tool = CodeAnalyzerTool()
    assert "Valid syntax" in code_tool.execute("def hello(): pass")
    assert "SyntaxError" in code_tool.execute("def broken(")


# 6. Eval Scorer Tests
def test_eval_scorer():
    scorer = EvalScorer()
    
    # Exact and numerical matches
    assert scorer.score_answer("The result is 42", "42") == 1.0
    assert scorer.score_answer("$45,000,000", "45000000") == 1.0
    assert scorer.score_answer("Wrong answer", "100") == 0.0

    # Tool precision/recall
    assert scorer.score_tools(["Calculator"], ["Calculator"]) == 1.0
    assert scorer.score_tools(["Calculator", "KnowledgeLookup"], ["Calculator"]) > 0.5
    assert scorer.score_tools([], ["Calculator"]) == 0.0

    # Trajectory efficiency
    assert scorer.score_trajectory(2, max_expected_steps=3) == 1.0
    assert scorer.score_trajectory(5, max_expected_steps=3) < 1.0


# 7. FastAPI Gateway Endpoints Tests
def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_chat_completions_endpoint(client):
    payload = {
        "messages": [{"role": "user", "content": "Hello world"}],
        "policy": "auto_router"
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "X-Model-Used" in response.headers
    assert "X-Cost-USD" in response.headers
    assert "X-Cache-Hit" in response.headers


def test_cache_hit_on_second_request(client):
    payload = {
        "messages": [{"role": "user", "content": "Unique prompt for caching test 12345"}],
        "policy": "auto_router"
    }
    # First call: cache miss
    res1 = client.post("/v1/chat/completions", json=payload)
    assert res1.status_code == 200
    assert res1.headers.get("X-Cache-Hit") == "false"

    # Second call: cache hit
    res2 = client.post("/v1/chat/completions", json=payload)
    assert res2.status_code == 200
    assert res2.headers.get("X-Cache-Hit") == "true"
    assert res2.headers.get("X-Cost-USD") == "0.0"


def test_logs_and_metrics_endpoints(client):
    logs_res = client.get("/v1/logs?limit=5")
    assert logs_res.status_code == 200
    assert "logs" in logs_res.json()

    metrics_res = client.get("/v1/metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert "total_requests" in metrics
    assert "total_cost_usd" in metrics


def test_dashboard_endpoint(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "Smart AI Gateway Dashboard" in resp.text


def test_evals_latest_endpoint(client):
    resp = client.get("/v1/evals/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "cost_savings_pct" in data
    assert "resume_bullet" in data
