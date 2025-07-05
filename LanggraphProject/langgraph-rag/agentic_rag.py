from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import os


class AgentState(TypedDict):
    query: str
    context: Optional[str]
    final_answer: Optional[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

retriever = None
if os.path.exists("vectorstore/index.faiss"):
    retriever = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True).as_retriever()

def retrieve_tool(query):
    if retriever is None:
        return "No documents indexed. Please upload and index a document first."
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def decide_to_retrieve(state: AgentState) -> str:
    if "data" in state["query"].lower() or "info" in state["query"].lower():
        return "retrieve"
    return "generate"

def retrieve_step(state: AgentState):
    context = retrieve_tool(state["query"])
    return {"context": context, "query": state["query"]}

def generate_answer(state: AgentState):
    prompt = f"Use this context to answer the question:\nContext:\n{state.get('context', '')}\n\nQuestion: {state['query']}"
    response = llm.invoke(prompt)
    print(response.content)
    return {"final_answer": response.content}

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_step)
graph.add_node("generate", generate_answer)
graph.set_entry_point("retrieve")

graph.add_conditional_edges("retrieve", decide_to_retrieve, {
    "retrieve": "retrieve",
    "generate": "generate"
})
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

def run_agent(query: str):
    result = app.invoke({"query": query})
    return result["final_answer"]
