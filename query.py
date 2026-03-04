from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ----------------------------
# Embeddings + DB
# ----------------------------
EMBEDDING_MODEL = "embeddinggemma"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="collection_1",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# ----------------------------
# Middleware RAG
# ----------------------------
from langchain.agents.middleware import dynamic_prompt, ModelRequest


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    last_user_message = request.state["messages"][-1].text

    # récupérer chunks pertinents
    retrieved_docs = vector_store.similarity_search(
        last_user_message,
        k=5
    )

    # afficher les chunks récupérés et source
    print("\n[INFO] Chunks récupérés :")
    for doc in retrieved_docs:
        print(f"- Source: {doc.metadata.get('source')}, Contenu (début): {doc.page_content[:150]}...\n")

    docs_content = "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n{doc.page_content[:800]}"
        for doc in retrieved_docs
    )

    system_message = (
        "You are a senior software engineer.\n"
        "Generate clear and structured technical documentation for the application.\n"
        "Use ONLY the provided context, do NOT invent anything.\n\n"
        f"Context:\n{docs_content}\n\n"
        "The documentation should include:\n"
        "- Overview\n"
        "- Main components\n"
        "- File structure / paths\n"
        "- API / services if applicable\n"
        "- Example usage if relevant\n"
        "Format in markdown.\n"
    )

    return system_message


# ----------------------------
# Agent RAG
# ----------------------------
from langchain.agents import create_agent
from langchain_ollama import OllamaLLM

model = OllamaLLM(
    model="qwen2.5:3b",
    temperature=0
)

agent = create_agent(
    model=model,
    tools=[],
    middleware=[prompt_with_context],
)

# ----------------------------
# Question utilisateur
# ----------------------------
query = "Génère la documentation complète de l'application Audacy : structure, chemins, modules, API, exemples."

for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
):
    step["messages"][-1].pretty_print()
