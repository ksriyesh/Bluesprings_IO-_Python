import os, json
from datetime import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, END

from tools.calculator_tool import calculator
from tools.translation_tool import translate
from tools.general_tool import general_assistant
from nodes.splitter import split_and_classify
from nodes.output_node import combine_and_format

# --- Log Directory + Session Log ---
SESSION_LOG = []
LOG_DIR = "integration_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Tool function mapping ---
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "translator": translate,
    "general": general_assistant,
}

# --- State schema for one sub-task ---
class AgentState(TypedDict, total=False):
    tool: str
    query: str
    result: str
    final_output: str

# --- Tool routing ---
def route_task(state: AgentState) -> str:
    return state.get("tool", END)

# --- Tool execution node ---
def tool_node(tool_name: str):
    def run(state: AgentState) -> AgentState:
        output = TOOL_FUNCTIONS[tool_name].invoke(state["query"])
        state["result"] = output
        return state
    return run

# --- Output formatting node ---
def output_node(state: AgentState) -> AgentState:
    tool = state.get("tool", "unknown")
    result_data = state.get("result", {})
    
    if isinstance(result_data, dict):
        response = result_data.get("response", "No response")
    else:
        response = result_data
    
    state["final_output"] = f"[{tool}] → {response}"
    return state

# --- Save full session log on exit ---
def log_full_session():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"session_log_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(SESSION_LOG, f, indent=2)

# --- LangGraph build (single sub-task execution) ---
builder = StateGraph(AgentState)

builder.add_node("calculator", tool_node("calculator"))
builder.add_node("translator", tool_node("translator"))
builder.add_node("general", tool_node("general"))
builder.add_node("output", output_node)

builder.set_entry_point("start")
builder.add_node("start", lambda state: state)
builder.add_conditional_edges("start", route_task)

builder.add_edge("calculator", "output")
builder.add_edge("translator", "output")
builder.add_edge("general", "output")

graph = builder.compile()

# --- CLI loop with grouped session logging ---
if __name__ == "__main__":
    print("💬 Agentic AI CLI — type 'exit' to quit.\n")

    while True:
        query = input("Ask me anything: ")

        if query.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting Agent. Saving full session log...")
            log_full_session()
            print("🗂️ Session log saved.")
            break

        subqueries = split_and_classify(query)
        all_results = []

        session_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": query,
            "subtasks": []
        }

        for task in subqueries:
            tool = task["tool"]
            subquery = task["query"]
            state = {"tool": tool, "query": subquery}

            # Invoke the tool via LangGraph
            result = graph.invoke(state)

            # Safely extract tool result and model
            tool_result = result.get("result", {})
            response = tool_result.get("response", str(tool_result)) if isinstance(tool_result, dict) else str(tool_result)
            model_used = tool_result.get("model", "unknown") if isinstance(tool_result, dict) else "unknown"
            final_output = f"[{tool}] → {response}"

            session_entry["subtasks"].append({
                "tool": tool,
                "query": subquery,
                "response": response,
                "model_used": model_used
            })

            all_results.append(final_output)

        SESSION_LOG.append(session_entry)

        print("\n🧠 Final Answer:")
        for res in all_results:
            print(res)
        print("\n" + "-" * 50 + "\n")
