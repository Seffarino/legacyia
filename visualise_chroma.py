# ----------------------------
# Permet de faire un plot pour visualiser la db chromaw
# ----------------------------


import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.decomposition import PCA

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ----------------------------
# Paramètres
# ----------------------------
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
PERSIST_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "collection_1"

OUTPUT_HTML = "embeddings_visualization.html"
CHUNKS_DIR = Path("chunks_html")
CHUNKS_DIR.mkdir(exist_ok=True)

# ----------------------------
# Charger Chroma
# ----------------------------
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

data = vector_store.get(include=["embeddings", "documents", "metadatas"])
embeds = np.array(data["embeddings"])
docs = data["documents"]
metas = data["metadatas"]

if len(embeds) == 0:
    raise RuntimeError("Aucun embedding trouvé dans Chroma")

# ----------------------------
# PCA 2D
# ----------------------------
vectors_2d = PCA(n_components=2).fit_transform(embeds)

# ----------------------------
# Générer pages HTML des chunks
# ----------------------------
links = []
for i, (doc, meta) in enumerate(zip(docs, metas)):
    html_path = CHUNKS_DIR / f"chunk_{i}.html"
    html_path.write_text(
        f"""
<html>
<body>
<h3>{meta.get("source", "Unknown")} – page {meta.get("page", "?")}</h3>
<pre>{doc}</pre>
</body>
</html>
""",
        encoding="utf-8"
    )
    links.append(str(html_path))

# ----------------------------
# DataFrame
# ----------------------------
df = pd.DataFrame({
    "PC1": vectors_2d[:, 0],
    "PC2": vectors_2d[:, 1],
    "source": [m.get("source", "Unknown") for m in metas],
    "page": [m.get("page", "?") for m in metas],
    "link": links,
})

# ----------------------------
# Plot Plotly
# ----------------------------
fig = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="source",
    hover_data=["page"],
    custom_data=["link"],
    title="Embeddings du rapport (PCA 2D)"
)
fig.update_layout(clickmode="event+select")

# ----------------------------
# Export HTML avec JS onclick
# ----------------------------
fig.write_html(
    OUTPUT_HTML,
    include_plotlyjs="cdn",
    full_html=True,
    post_script="""
var plot = document.getElementsByClassName('plotly-graph-div')[0];
plot.on('plotly_click', function(data){
    var link = data.points[0].customdata[0];
    window.open(link, '_blank');
});
"""
)

print(f"[OK] Visualisation générée : {OUTPUT_HTML}")
