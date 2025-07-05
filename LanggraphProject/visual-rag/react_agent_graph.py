# react_agent_graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Literal
from LLM.gemini import ask_gemini_question

# Define state passed between steps
class AgentState(TypedDict):
    image_path: str
    question: str
    final_answer: Optional[str]
    status: Literal["CONTINUE", "FINISH"]
    attempts: int  

# Main action
def react_step(state: AgentState) -> AgentState:
    print("Running react_step...")
    print("Current state:", state)

    image_path = state["image_path"]
    question = state["question"]
    attempts = state.get("attempts", 0)

    # Step 1: Call Gemini
    answer = ask_gemini_question(image_path, question)
    print("Gemini Answer:", answer)

    # Step 2: Clean the answer
    cleaned = (answer or "").strip().lower()

    # Step 3: Define vague answers that mean "I don't know"
    vague_responses = ["", "i'm not sure", "not sure", "cannot determine", "unclear", "not visible", "i can't tell","I'm only able to answer questions about the image content."]

    # Step 4: Retry logic: try up to 3 times
    if cleaned in vague_responses:
        if attempts >= 2:  # i.e., this is the 3rd try
            return {
                **state,
                "final_answer": "I'm not sure based on the image.",
                "status": "FINISH",
                "attempts": attempts + 1
            }
        else:
            print("Vague response — retrying react_step.")
            return {
                **state,
                "final_answer": None,
                "status": "CONTINUE",
                "attempts": attempts + 1
            }

    # Step 5: If answer is valid, mark as FINISH
    return {
        **state,
        "final_answer": answer.strip(),
        "status": "FINISH",
        "attempts": attempts + 1
    }


# Control flow
def route(state: AgentState) -> str:
    print("Routing decision...")
    print("Current state:", state)
    return "FINISH" if state.get("final_answer") else "CONTINUE"

# Build graph
def build_react_graph():
    builder = StateGraph(AgentState)
    builder.add_node("react_step", react_step)
    builder.set_entry_point("react_step")
    builder.add_conditional_edges("react_step", route, {
        "FINISH": END,
        "CONTINUE": "react_step"
    })
    return builder.compile()
