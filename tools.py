"""Custom tools for the LangGraph agent demo."""

import math
import datetime
import json
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic, powers, and math functions.

    Args:
        expression: A mathematical expression string (e.g., "2 + 3 * 4", "sqrt(16)", "2**10")
    """
    # Safe evaluation with limited builtins
    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }
    safe_locals = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}

    try:
        result = eval(expression, safe_globals, safe_locals)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: Timezone name (currently only 'UTC' and 'local' are supported)
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if timezone.lower() == "local":
        now_local = datetime.datetime.now()
        return f"Local time: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')})"
    return f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock data for demo purposes).

    Args:
        city: The name of the city to get weather for
    """
    # Mock weather data for demo
    mock_weather = {
        "tokyo": {"temp": 22, "condition": "Partly Cloudy", "humidity": 65, "wind": "12 km/h NE"},
        "new york": {"temp": 15, "condition": "Sunny", "humidity": 45, "wind": "8 km/h W"},
        "london": {"temp": 10, "condition": "Overcast", "humidity": 80, "wind": "20 km/h SW"},
        "sydney": {"temp": 28, "condition": "Clear", "humidity": 55, "wind": "15 km/h SE"},
        "paris": {"temp": 12, "condition": "Light Rain", "humidity": 75, "wind": "18 km/h NW"},
    }

    city_lower = city.lower()
    if city_lower in mock_weather:
        w = mock_weather[city_lower]
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature: {w['temp']}°C\n"
            f"  Condition: {w['condition']}\n"
            f"  Humidity: {w['humidity']}%\n"
            f"  Wind: {w['wind']}"
        )
    return f"Weather data not available for '{city}'. Available cities: {', '.join(c.title() for c in mock_weather)}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search a knowledge base for information on various topics (mock data for demo purposes).

    Args:
        query: The search query to look up
    """
    # Mock knowledge base
    kb = {
        "langchain": (
            "LangChain is a framework for developing applications powered by large language models (LLMs). "
            "It provides tools for chaining LLM calls, managing prompts, integrating with external data sources, "
            "and building agents. Key components include Chains, Agents, Memory, and Retrievers."
        ),
        "langgraph": (
            "LangGraph is a library built on top of LangChain for creating stateful, multi-actor applications "
            "with LLMs. It uses a graph-based approach where nodes represent computation steps and edges define "
            "the flow between them. It excels at building complex agentic workflows and multi-agent systems."
        ),
        "aws bedrock": (
            "Amazon Bedrock is a fully managed service that makes foundation models (FMs) from leading AI companies "
            "available via an API. It supports models from Anthropic (Claude), Meta (Llama), Mistral, AI21, Cohere, "
            "and Amazon (Titan). Bedrock offers features like model fine-tuning, RAG with Knowledge Bases, and Agents."
        ),
        "inference profile": (
            "AWS Bedrock Application Inference Profiles (AIPs) are configurations that route requests to specific "
            "foundation models with predefined settings. They allow teams to share model configurations, enforce "
            "usage policies, and track costs separately. AIPs are identified by ARNs and can be used as model IDs "
            "in API calls."
        ),
        "react agent": (
            "ReAct (Reasoning + Acting) is an agent pattern where the LLM iteratively reasons about a task and "
            "takes actions using available tools. The agent follows a Thought → Action → Observation loop until "
            "it reaches a final answer. LangGraph implements this pattern with explicit state management."
        ),
    }

    query_lower = query.lower()
    results = []
    for key, value in kb.items():
        if any(word in key for word in query_lower.split()) or any(word in query_lower for word in key.split()):
            results.append(f"[{key.title()}]\n{value}")

    if results:
        return "\n\n".join(results)
    return f"No results found for '{query}'. Try searching for: LangChain, LangGraph, AWS Bedrock, Inference Profile, or ReAct Agent."


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units of measurement.

    Args:
        value: The numeric value to convert
        from_unit: The source unit (e.g., 'km', 'miles', 'kg', 'lbs', 'celsius', 'fahrenheit')
        to_unit: The target unit
    """
    conversions = {
        # Distance
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v / 3.28084,
        # Weight
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v / 2.20462,
        # Temperature
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "kelvin"): lambda v: v + 273.15,
        ("kelvin", "celsius"): lambda v: v - 273.15,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return (
        f"Conversion from '{from_unit}' to '{to_unit}' not supported. "
        f"Supported conversions: km↔miles, m↔ft, kg↔lbs, celsius↔fahrenheit, celsius↔kelvin"
    )
