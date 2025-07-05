from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from tools import createRagTool,createSearchTool,createWeatherTool,createAddTool,createMultiplyTool
from langchain_google_genai import  ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import os

# define llm
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# create tools to bind to llm
search_tool=createSearchTool()
weather_info_tool=createWeatherTool()
rag_tool = createRagTool()
addTool = createAddTool
multiplyTool = createMultiplyTool

print(rag_tool)
print(addTool)
print(multiplyTool)


tools = [search_tool, weather_info_tool,rag_tool,addTool,multiplyTool]
llm_with_tools = llm.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")

app = builder.compile()

def run_agent(query: str):
    messages = [HumanMessage(content=query)]
    result = app.invoke({"messages": messages})
    return result['messages'][-1].content


