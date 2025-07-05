import os
import random
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


retriever = None
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def load_and_index(file_path):
    print("loaded file")
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")

    docs = loader.load()
    print("splitting into chunks ....")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print("creating embeddings into chunks ....")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vectorstore")
    print("saved to vectorestore")



def retrieve_tool(query:str)->str:
    print("Inside retrieve_tool")
    """Retrieves detailed information about gala guests based on their name or relation."""

    if os.path.exists("vectorstore/index.faiss"):
        retriever = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True).as_retriever()
    if retriever is None:
        return "No documents indexed. Please upload and index a document first."
    docs = retriever.invoke(query)
    if docs:
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "No matching guest information found."
    
## additional tool definitions
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"





