from langchain_google_genai import  ChatGoogleGenerativeAI
import os
from typing import Annotated,Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()



llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class MessageClassifier(BaseModel):
    message_type: Literal["emotional","logical"] = Field (
        description = "Classify if the message requires an emotional (therapists) or logical response"

    )

class State(TypedDict):
    messages : Annotated[list,add_messages]
    message_type: str | None

def classify_message(state:State):
    print("Inside classify agent")
    last_msg = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    sys_prompt = """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
    result = classifier_llm.invoke([SystemMessage(content=sys_prompt),last_msg])
    print("Classified ur msg as", result.message_type)
    return {"message_type":result.message_type}

def router(state:State):
    print("Inside router agent")
    message_type = state.get("message_type","logical")
    if message_type == "emotional":
        return {"next":"therapist"}
    return {"next":"logical"}

def therapist_agent(state:State):
    print("Inside therapist agent")
    last_msg = state["messages"][-1]
    message = [
        SystemMessage(content="""You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""),

        last_msg]
    reply = llm.invoke(message)
    return {"messages":[AIMessage(content=reply.content)]}

def logical_agent(state:State):
    print("Inside logical agent")
    last_msg = state["messages"][-1]
    message = [
        SystemMessage(content="""You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""),
        last_msg]
    reply = llm.invoke(message)

    return {"messages":[AIMessage(content=reply.content)]}



graph_builder = StateGraph(State)
graph_builder.add_node("Classifier",classify_message)
graph_builder.add_node("Router",router)
graph_builder.add_node("Therapist",therapist_agent)
graph_builder.add_node("Logic_Buff",logical_agent)

graph_builder.add_edge(START,"Classifier")
graph_builder.add_edge("Classifier","Router")
graph_builder.add_conditional_edges(
    "Router",
    lambda state: state.get("next"),
    {"therapist": "Therapist", "logical": "Logic_Buff"}
)
graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            HumanMessage(content=user_input)
        ]
        print("before",state)
        state = graph.invoke(state)
        print("after",state)
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()


