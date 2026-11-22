# ⚡ Smart AI Gateway: LLMOps + Agent Evals + Auto-Routing

A production-grade backend AI gateway service that sits between client applications/agents and LLM providers. It brings together the three core disciplines AI engineering teams actively hire for: **Auto-Routing**, **LLMOps (Observability, Cost Tracking, Resilience, Caching)**, and **Agent Evaluations**.

---

## 🎯 What Goes On Your Resume

> *"Built an LLM gateway with auto-routing, fallback and cost tracking; auto-routing cut LLM cost by 69.9% while keeping agent eval pass rate within 16.0 points of always using the strongest model."*

---

## 🍽️ The 3 Core Topics in Simple Terms (The Restaurant Analogy)

| Topic | What it Means | Restaurant Analogy |
|---|---|---|
| **Auto-Routing** | Send each request to the right model: easy tasks to a cheap, fast model; hard tasks to a strong, high-reasoning one. | *Manager decides whether the junior cook or head chef cooks each dish.* |
| **LLMOps** | Log every request (model, tokens, latency, cost), cache repeated prompts, retry with backoff, and fall back when a provider fails. | *Manager’s order book, billing ledger, and backup cook on standby.* |
| **Agent Evals** | Test whether an autonomous AI agent solved the task right: the final answer, the tools selected, and the step trajectory. | *The food taster tasting every dish before it leaves the kitchen.* |

---

## 🏗️ Architecture

```
                                 [ Client Apps / Agents ]
                                            │
                                ┌───────────▼───────────┐
                                │ POST /v1/chat/completions
                                ├───────────────────────┤
                                │    AI GATEWAY CORE    │
                                └───────────┬───────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
             [ Cache Check ]                               [ Smart Router ]
       (SHA-256 Key Normalization)                  (Complexity Heuristics: 0.0-1.0)
                    │                                               │
           HIT ─────┴───── MISS                       ┌─────────────┴─────────────┐
        (0ms, $0)                                     │                           │
                                                Score < 0.30                Score >= 0.30
                                                      │                           │
                                            ┌─────────▼─────────┐       ┌─────────▼─────────┐
                                            │    CHEAP MODEL    │       │   STRONG MODEL    │
                                            │(LLaMA-3.1-8B etc.)│       │(LLaMA-3.3-70B etc)│
                                            └─────────┬─────────┘       └─────────┬─────────┘
                                                      │                           │
                                                      └─────────► Fail / 429 ─────┘
                                                                (Auto Fallback)
                                                                      │
                                                            ┌─────────▼─────────┐
                                                            │  LLMOps Storage   │
                                                            │ (SQLite Telemetry)│
                                                            └─────────┬─────────┘
                                                                      │
                                                        ┌─────────────┴─────────────┐
                                                        │                           │
                                                 [ Web Dashboard ]          [ Eval Benchmark ]
                                                 (Live Metrics & UI)       (25 Tasks x 3 Policies)
```

---

## 📊 Evaluation Benchmark Results (Real Numbers)

We tested an autonomous agent with 3 tools (`Calculator`, `KnowledgeLookup`, `CodeAnalyzer`) across a standardized exam of 25 tasks under 3 policies:
1. **ALWAYS_CHEAP**: Every call is forced through the cheap, fast model (`llama-3.1-8b-instant`).
2. **ALWAYS_STRONG**: Every call is forced through the large, high-reasoning model (`llama-3.3-70b-versatile`).
3. **AUTO_ROUTER**: The gateway's complexity engine evaluates prompt heuristics in real time and routes easy tasks to cheap and hard tasks to strong.

### Summary Comparison Table

| Policy | Pass Rate (%) | Total Cost (25 Tasks) | Avg Latency (ms) | Speedup vs Strong | Cost Savings vs Strong |
|---|:---:|:---:|:---:|:---:|:---:|
| **ALWAYS_CHEAP** | 60.0% | $0.000671 | 73.8 ms | **3.9x Faster** | 94.0% cheaper |
| **ALWAYS_STRONG** | **88.0%** | $0.011108 | 287.5 ms | Baseline (1.0x) | Baseline |
| **AUTO_ROUTER** | **72.0%** | **$0.003338** | **123.2 ms** | **2.3x Faster** | **69.9% CHEAPER** |

*Key Takeaway: Auto-routing cut LLM token costs by **69.9%** and reduced average latency by **57%** while maintaining a high evaluation pass rate within 16 points of the strongest model.*

---

## 🛠️ Tech Stack

- **API Framework**: Python 3.11+, FastAPI (async), Uvicorn
- **HTTP & Resilience**: HTTPX (async), Exponential Backoff with Jitter, Automatic Fallback
- **Models Supported**: Groq (Free tier LLaMA 3.1/3.3), OpenAI (`gpt-4o`, `gpt-4o-mini`), any OpenAI-compatible API, and an integrated **Mock Provider** (for offline testing & reproducible CI runs)
- **Database & Storage**: SQLite via `aiosqlite` (async connection pooling)
- **Caching**: In-memory `TTLCache` with SHA-256 key hashing
- **Agent Framework**: Custom ReAct Agent with 3 tools (`Calculator`, `KnowledgeLookup`, `CodeAnalyzer`)
- **Evaluation Suite**: Multi-dimensional scoring (Answer Accuracy, Tool Precision/Recall, Trajectory Steps)
- **Visualization**: Chart.js (Interactive Web Dashboard) & Matplotlib (Publication-quality PNG charts)
- **Shipping**: Docker, Docker Compose, GitHub Actions CI

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
cd d:/moni/ai-gateway
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional) If you have a Groq or OpenAI API key, set `GROQ_API_KEY` or `OPENAI_API_KEY` in `.env`. If left empty, the built-in deterministic Mock Provider runs automatically.*

### 3. Run the Test Suite

```bash
pytest -v
```

### 4. Run the 3-Policy Evaluation Benchmark

```bash
python evals/benchmark_runner.py
```
This executes the 25-task exam under all 3 policies, saves the side-by-side plot to `benchmark_results.png`, and prints your custom resume bullet.

### 5. Launch the AI Gateway & Live Dashboard

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Dashboard**: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Healthcheck**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🚢 Docker Deployment

```bash
# Build and run container
docker compose up -d

# Check health
curl http://localhost:8000/health
```

---

## 🎙️ Interview Preparation Guide (4 Key Questions)

### Q1: How does your router decide which model gets a request?
> **Answer**: The router evaluates incoming requests using transparent, explainable heuristic dimensions to assign a `complexity_score` between `0.0` and `1.0`:
> 1. **Length & Context Density**: Prompts with high word/token counts (>120 or >300 words) receive higher weights.
> 2. **Code & Syntax Presence**: Regex detection of code blocks, function signatures (`def`, `class`), and SQL queries adds up to +0.40.
> 3. **High-Order Reasoning Keywords**: Matches on keywords like `step-by-step`, `prove`, `refactor`, `optimize`, `derivation`, `time complexity` add up to +0.35.
> 4. **Tool/Agent Schema Complexity**: Presence of tool specifications adds +0.25.
> 5. **Direct Intent Discount**: Trivial greetings or short factual queries subtract score.
>
> If the `complexity_score >= 0.30`, the request is routed to the **Strong Model**; otherwise it is routed to the **Cheap Model**. Every routing decision logs an explicit human-readable `routing_reason` into SQLite for auditability.

### Q2: How do you evaluate an autonomous agent, not just a single LLM answer?
> **Answer**: Evaluating an agent requires evaluating the full execution trajectory, not just the output string:
> 1. **Answer Correctness (50% weight)**: Compares final answer against ground truth with numerical tolerance and substring extraction.
> 2. **Tool Selection Correctness (30% weight)**: Calculates the precision, recall, and F1-score of the tools invoked versus the expected toolset (e.g. did it call `KnowledgeLookup` and then `Calculator` without hallucinating or using redundant tools?).
> 3. **Trajectory & Step Efficiency (20% weight)**: Checks whether the agent solved the task within the expected number of steps, penalizing looping or excessive backtracking.
>
> Composite score `>= 0.70` marks the task as passed.

### Q3: How do you calculate LLM costs, and what happens when a provider times out or rate-limits?
> **Answer**:
> - **Cost Calculation**: We maintain an in-memory pricing table specifying micro-dollar rates per 1,000,000 prompt and completion tokens per model tier. For every completion, we calculate:
>   $$\text{Cost} = \left(\frac{\text{Prompt Tokens}}{10^6} \times \text{Rate}_{\text{in}}\right) + \left(\frac{\text{Completion Tokens}}{10^6} \times \text{Rate}_{\text{out}}\right)$$
>   Cached hits return `0.0ms` latency and `$0.000000` cost.
> - **Rate Limits & Timeouts**: The gateway implements an async retry loop with exponential backoff and randomized jitter for transient 429 and 5xx errors. If retries are exhausted, the gateway triggers **Automatic Fallback** to a secondary model/provider, seamlessly serving the client and logging `fallback_used: true`.

### Q4: Why not just use LiteLLM or Portkey?
> **Answer**: While tools like LiteLLM and Portkey are excellent off-the-shelf proxies, building the gateway from scratch gives complete architectural ownership over:
> 1. Custom domain-specific routing rules tailored to our agent's exact toolsets.
> 2. Zero-latency in-process response caching without external SaaS dependencies.
> 3. A native Agent Evaluation harness directly validating that our routing threshold does not degrade production pass rates.

---

## 📁 Repository Structure

```
ai-gateway/
├── config.py                  # Pydantic settings, model pricing tables, and policies
├── cost_engine.py             # Precise token-to-dollar micro-cost calculator
├── storage.py                 # Async SQLite database and LLMOps analytics queries
├── cache.py                   # In-memory TTLCache with SHA-256 key hashing
├── providers.py               # OpenAI/Groq API client + Mock Provider with error injection
├── router.py                  # Explainable Complexity Router & Policy Engine
├── main.py                    # FastAPI application, OpenAPI endpoints & HTML dashboard
├── agent.py                   # Autonomous ReAct Agent with 3 tools
├── evals/
│   ├── dataset.py             # Standardized 25-task benchmark exam
│   ├── scorer.py              # Multi-dimensional evaluation scorer
│   └── benchmark_runner.py    # 3-policy evaluation harness, plots & resume bullet
├── tests/
│   └── test_gateway.py        # Comprehensive test suite (11 unit/integration tests)
├── .github/workflows/ci.yml   # GitHub Actions CI workflow (lint, test, evals)
├── Dockerfile                 # Production Docker container
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Python dependencies
└── README.md                  # Complete documentation and interview guide
```
