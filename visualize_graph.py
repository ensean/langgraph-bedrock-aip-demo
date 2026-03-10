"""
Visualize the LangGraph agent graph structure.
Prints an ASCII representation and optionally saves a PNG.
"""

from agent import build_graph


def ascii_graph() -> None:
    """Print an ASCII diagram of the agent graph."""
    print("""
    LangGraph Agent — Graph Structure
    ══════════════════════════════════════

         ┌─────────────────┐
         │     START       │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │                 │◄──────────────┐
         │     agent       │               │
         │  (LLM + tools)  │               │
         │                 │               │
         └────────┬────────┘               │
                  │                        │
          tools_condition                  │
         /                \\               │
        ▼                  ▼              │
 ┌──────────┐         ┌─────────┐         │
 │  tools   │─────────►         │         │
 │  (node)  │         │   END   │         │
 └──────────┘         └─────────┘         │
        │                                 │
        └─────────────────────────────────┘
          (loop back to agent after tool execution)

    Nodes:
      • agent  — Calls the LLM (Bedrock AIP). If the model requests
                 tool calls, routes to the tools node. Otherwise ends.
      • tools  — Executes the requested tool(s) and returns results
                 to the agent node.

    Edges:
      START → agent
      agent → tools   (when tool_calls present in response)
      agent → END     (when no tool_calls — final answer)
      tools → agent   (always loops back for next reasoning step)

    Available tools:
      • calculator          — evaluate math expressions
      • get_current_datetime — current date/time
      • get_weather          — city weather (mock)
      • search_knowledge_base — tech knowledge lookup
      • unit_converter       — unit conversions
    """)


def save_png() -> None:
    """Save a PNG visualization of the graph (requires graphviz)."""
    try:
        from IPython.display import Image  # noqa: PLC0415

        agent = build_graph()
        png_data = agent.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved to graph.png")
    except Exception as e:
        print(f"Could not save PNG (graphviz may not be installed): {e}")
        print("Run the ASCII visualization instead.")


if __name__ == "__main__":
    import sys

    ascii_graph()

    if len(sys.argv) > 1 and sys.argv[1] == "--png":
        save_png()
