"""
LangGraph Agentic Demo with AWS Bedrock Application Inference Profile
— API Key Authentication variant

Authenticates using a Bedrock API key (Authorization: Bearer <key>)
instead of IAM AK/SK credentials.

Usage:
    export BEDROCK_API_KEY="your-bedrock-api-key"
    python agent_apikey.py          # interactive mode
    python agent_apikey.py --demo   # automated demo queries
"""

import os
import sys
from typing import Annotated

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from tools import calculator, get_current_datetime, get_weather, search_knowledge_base, unit_converter

# ── Configuration ─────────────────────────────────────────────────────────────

# TODO: Replace the AWS account ID (123456789012) with your own account ID
AIP_ARN = "arn:aws:bedrock:ap-northeast-1:123456789012:application-inference-profile/qs2x0oirm4py"

TOOLS = [calculator, get_current_datetime, get_weather, search_knowledge_base, unit_converter]

SYSTEM_PROMPT = """You are a helpful research and data assistant powered by AWS Bedrock.
You have access to the following tools:

1. **calculator** – evaluate mathematical expressions
2. **get_current_datetime** – get the current date and time
3. **get_weather** – look up weather for a city
4. **search_knowledge_base** – search for information on tech topics
5. **unit_converter** – convert between units of measurement

Think step-by-step. Use tools whenever they would help you give a more accurate or complete answer.
After gathering information, synthesize a clear and concise response."""


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed between nodes in the LangGraph graph."""
    messages: Annotated[list[BaseMessage], add_messages]


# ── Build the LLM ─────────────────────────────────────────────────────────────

def build_llm(api_key: str) -> ChatBedrockConverse:
    """
    Create a ChatBedrockConverse instance authenticated via Bedrock API key.

    AWS Bedrock API keys are sent as:  Authorization: Bearer <key>
    boto3 SigV4 signing is disabled (UNSIGNED) so the header isn't overwritten.
    """
    # Disable SigV4 signing; dummy credentials prevent boto3 from
    # searching the credential chain and raising NoCredentialsError.
    session = boto3.Session(
        aws_access_key_id="placeholder",
        aws_secret_access_key="placeholder",
        region_name="ap-northeast-1",
    )
    bedrock_client = session.client(
        "bedrock-runtime",
        config=Config(signature_version=UNSIGNED),
    )

    # Inject the Bearer token just before each HTTP request is sent.
    def _inject_bearer(request, **kwargs):
        request.headers["Authorization"] = f"Bearer {api_key}"

    bedrock_client.meta.events.register("before-send.bedrock-runtime.*", _inject_bearer)

    return ChatBedrockConverse(
        model=AIP_ARN,
        provider="anthropic",
        client=bedrock_client,
        temperature=0.3,
        max_tokens=2048,
    )


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def build_graph(api_key: str) -> StateGraph:
    """Build and return the compiled LangGraph agent graph."""
    llm = build_llm(api_key)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(messages, config)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Interactive CLI ────────────────────────────────────────────────────────────

def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def run_interactive(agent) -> None:
    print_separator("═")
    print("  LangGraph Agentic Demo — AWS Bedrock API Key Authentication")
    print_separator("═")
    print(f"  Model ARN : {AIP_ARN}")
    print(f"  Auth      : Bearer token (BEDROCK_API_KEY)")
    print(f"  Tools     : {', '.join(t.name for t in TOOLS)}")
    print_separator()
    print("  Type your question and press Enter. Type 'quit' or 'exit' to stop.")
    print_separator()

    conversation_history: list[BaseMessage] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        conversation_history.append(HumanMessage(content=user_input))
        print("\nAgent thinking...\n")
        print_separator("-")

        full_response = ""
        tool_calls_made = []

        for step in agent.stream(
            {"messages": conversation_history},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            if isinstance(last_msg, AIMessage):
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        if tc not in tool_calls_made:
                            tool_calls_made.append(tc)
                            print(f"[Tool] Calling: {tc['name']}({tc['args']})")
                else:
                    full_response = last_msg.content

        print_separator("-")
        print(f"\nAssistant: {full_response}\n")
        conversation_history.append(AIMessage(content=full_response))


def run_demo_queries(agent) -> None:
    demo_queries = [
        "What is the square root of 144 multiplied by the circumference constant pi, rounded to 4 decimal places?",
        "What's the weather like in Tokyo and London today? Which city is warmer?",
        "Convert 100 miles to kilometers, and also convert 37 degrees Celsius to Fahrenheit.",
        "What is LangGraph and how does it relate to AWS Bedrock inference profiles?",
        "What is today's date and time in UTC? Also calculate how many seconds are in a week.",
    ]

    print_separator("═")
    print("  LangGraph Agentic Demo — Automated Query Showcase (API Key Auth)")
    print_separator("═")
    print(f"  Model ARN : {AIP_ARN}")
    print(f"  Auth      : Bearer token (BEDROCK_API_KEY)")
    print(f"  Tools     : {', '.join(t.name for t in TOOLS)}")
    print_separator()

    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Query {i}/{len(demo_queries)}] {query}")
        print_separator("-")

        tool_calls_made = []
        final_answer = ""

        for step in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            if isinstance(last_msg, AIMessage):
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        if tc not in tool_calls_made:
                            tool_calls_made.append(tc)
                            print(f"  [Tool] {tc['name']}({tc['args']})")
                else:
                    final_answer = last_msg.content

        print(f"\n  Answer: {final_answer}")
        print_separator()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("BEDROCK_API_KEY", "").strip()
    if not api_key:
        print("Error: environment variable BEDROCK_API_KEY is not set.")
        print("  export BEDROCK_API_KEY=\"your-bedrock-api-key\"")
        sys.exit(1)

    agent = build_graph(api_key)

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo_queries(agent)
    else:
        run_interactive(agent)
