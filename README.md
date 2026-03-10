# LangGraph Agentic Demo — AWS Bedrock Application Inference Profile

A ReAct-style agentic demo built with **LangChain** + **LangGraph**, using an AWS Bedrock **Application Inference Profile (AIP)** as the LLM backend.

## Architecture

```
START → agent ──(has tool calls?)──► tools ──► agent (loop)
                └──(no tool calls)──► END
```

The agent follows a **Thought → Action → Observation** loop:
1. The LLM reasons about the query and decides which tool to call
2. The tool node executes the requested tool(s)
3. Results are fed back to the LLM for the next reasoning step
4. The loop continues until the LLM produces a final answer

## Files

| File | Description |
|------|-------------|
| `agent.py` | Main agent graph, LLM setup, interactive CLI |
| `tools.py` | Custom LangChain tools |
| `visualize_graph.py` | ASCII + optional PNG graph diagram |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure AWS credentials

The demo uses your default AWS credential chain. Set one of:

```bash
# Option A: Environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...   # if using temporary credentials

# Option B: AWS profile
export AWS_PROFILE=my-profile

# Option C: IAM role (on EC2/Lambda/ECS — automatic)
```

The Bedrock client is set to `ap-northeast-1` (Tokyo) to match the AIP region.

## Usage

### Interactive mode (multi-turn chat)

```bash
source .venv/bin/activate
python agent.py
```

### Automated demo queries

```bash
python agent.py --demo
```

### Visualize the graph

```bash
python visualize_graph.py
```

## Application Inference Profile

| Setting | Value |
|---------|-------|
| ARN | `arn:aws:bedrock:ap-northeast-1:123456789012:application-inference-profile/qs2x0oirm4py` |
| Region | `ap-northeast-1` |

The AIP ARN is passed directly as `model=AIP_ARN` to `ChatBedrockConverse`, which handles routing to the underlying foundation model.

## Available Tools

| Tool | Description |
|------|-------------|
| `calculator` | Evaluate math expressions (supports `math` functions like `sqrt`, `log`, `pi`) |
| `get_current_datetime` | Get current UTC or local date/time |
| `get_weather` | Mock weather data for major cities |
| `search_knowledge_base` | Mock KB with info on LangChain, LangGraph, Bedrock, etc. |
| `unit_converter` | Convert km↔miles, kg↔lbs, Celsius↔Fahrenheit, etc. |

## Example Queries

```
What is sqrt(256) + log(100)?
What's the weather in Tokyo and Paris? Which is warmer?
Convert 100 miles to km and 30°C to Fahrenheit.
What is LangGraph and how does it differ from LangChain?
What time is it in UTC right now?
```
