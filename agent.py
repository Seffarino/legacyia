from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma

from pathlib import Path
import gradio as gr

# ----------------------------
# Config Chroma / embeddings
# ----------------------------
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "collection_1"

Path(PERSIST_DIR).mkdir(exist_ok=True)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# ----------------------------
# Outil RAG (Agent Tool)
# ----------------------------
@tool
def rag_search(query: str) -> str:
    """
    Search relevant context from the knowledge base.
    Use this tool when factual or contextual information is needed.
    """
    docs = vector_store.similarity_search(query, k=5)
    return "\n\n".join(doc.page_content[:1500] for doc in docs)

# ----------------------------
# Modèle
# ----------------------------
model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

# ----------------------------
# Prompt agent
# ----------------------------
SYSTEM_PROMPT = """
You are an expert developer assistant.

You can reason step by step internally.
When external knowledge is required, use the available tools.
Use tool outputs to improve your final answer.
You can find the report in the context

reveal your internal reasoning.
"""

# ----------------------------
# Création de l'agent
# ----------------------------
agent = create_agent(
    model=model,
    tools=[rag_search],
    system_prompt=SYSTEM_PROMPT,
)

# ----------------------------
# Fonction Gradio
# ----------------------------
def chat_fn(message, history):
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": message}
            ]
        }
    )
    return result["messages"][-1].content

# ----------------------------
# UI Gradio
# ----------------------------
ui = gr.ChatInterface(
    fn=chat_fn,
    title="Agentic RAG (LangChain + Ollama)",
    description="Chat avec un agent LangChain utilisant RAG + outils",
    examples=[
        "What is this application about?",
        "Give me the list of examples provided in this app",
        "Explain the architecture",
    ],
)

# ----------------------------
# Lancement
# ----------------------------
if __name__ == "__main__":
    ui.launch()
