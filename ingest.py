import warnings
from pathlib import Path
from tqdm import tqdm
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

warnings.filterwarnings("ignore")

# ----------------------------
# Paramètres
# ----------------------------
FOLDER = Path("./legacy_code")
BATCH_SIZE = 50

CHUNK_SIZE_TEXT = 1000
CHUNK_SIZE_CODE = 800

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "collection_1"

# ----------------------------
# Utils
# ----------------------------
def clean_text(text: str) -> str:
    return re.sub(r"[\x00-\x1F\x7F]+", " ", text).strip()

def detect_language(suffix: str):
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".xml": "xml",
        ".sql": "sql",
        ".sh": "bash"
    }
    return mapping.get(suffix.lower(), "unknown")

# ----------------------------
# Chargement PDF + Code
# ----------------------------
all_docs = []

CODE_EXTENSIONS = [
    "*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c",
    "*.cs", "*.go", "*.rb", "*.php", "*.html",
    "*.css", "*.json", "*.xml", "*.sql", "*.sh"
]

# ---- PDF ----
for f in tqdm(FOLDER.rglob("*.pdf"), desc="Loading PDF"):
    try:
        loader = PyPDFLoader(str(f))
        docs = loader.load()

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update({
                "source": str(f),
                "type": "pdf"
            })

        all_docs.extend(docs)

    except Exception as e:
        print(f"[SKIP PDF] {f}: {e}")

# ---- CODE ----
for ext in CODE_EXTENSIONS:
    for f in tqdm(FOLDER.rglob(ext), desc=f"Loading {ext}"):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            content = clean_text(content)

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(f),
                    "filename": f.name,
                    "extension": f.suffix,
                    "language": detect_language(f.suffix),
                    "type": "code"
                }
            )

            all_docs.append(doc)

        except Exception as e:
            print(f"[SKIP CODE] {f}: {e}")

print(f"[INFO] Documents chargés : {len(all_docs)}")

if not all_docs:
    print("[ERROR] Aucun document chargé.")
    exit(1)

# ----------------------------
# Splitters spécialisés
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_TEXT,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "]
)

code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CODE,
    chunk_overlap=100,
    separators=[
        "\nclass ",
        "\ndef ",
        "\nfunction ",
        "\npublic ",
        "\nprivate ",
        "\nprotected ",
        "\n\n",
        "\n",
        " "
    ]
)

split_docs = []

for doc in tqdm(all_docs, desc="Splitting documents"):
    if doc.metadata.get("type") == "code":
        chunks = code_splitter.split_documents([doc])
    else:
        chunks = text_splitter.split_documents([doc])

    split_docs.extend(chunks)

print(f"[INFO] Total chunks : {len(split_docs)}")

if not split_docs:
    print("[ERROR] Aucun chunk généré.")
    exit(1)

# ----------------------------
# Embedding + Chroma
# ----------------------------
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

for i in tqdm(range(0, len(split_docs), BATCH_SIZE), desc="Embedding"):
    vector_store.add_documents(split_docs[i:i + BATCH_SIZE])

print(f"[DONE] Ingestion terminée : {len(split_docs)} chunks.")