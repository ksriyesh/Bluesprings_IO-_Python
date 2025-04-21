# nodes/output_node.py
def combine_and_format(results: list) -> str:
    """
    Merge outputs from all tools with their step traces.
    """
    formatted = []
    for res in results:
        formatted.append(f"[{res['tool']}] → {res['output']}")
    return "\n".join(formatted)