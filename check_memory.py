import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
mem = client.get_collection("user_memory", embedding_function=ef)

everything = mem.get()
for doc in everything["documents"][-3:]:
    print(doc)
    print("---")