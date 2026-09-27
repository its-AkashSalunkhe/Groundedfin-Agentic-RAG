'''Next step: chunk and load into Chroma (talking about the BE(A)WARE booklet and RBI press releases).'''



import os, re, json
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

def chunk_booklet(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    # split on numbered section headers like "1. Phishing links"
    pattern = re.compile(r'\n\s*(\d{1,2})\.\s+([A-Z][^\n]+)')
    matches = list(pattern.finditer(full_text))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        title = m.group(2).strip()
        body = full_text[start:end].strip()
        if len(body) > 80:  # skip stray table-of-contents matches
            chunks.append({"title": title, "text": body, "source": "BE(A)WARE booklet"})
    return chunks

def chunk_press_releases(folder):
    chunks = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                text = f.read()
            title = text.split("\n")[1] if len(text.split("\n")) > 1 else fname
            chunks.append({"title": title, "text": text, "source": f"RBI press release ({fname})"})
    return chunks

if __name__ == "__main__":
    booklet_chunks = chunk_booklet("data/rbi_docs/BEAWARE_booklet.pdf")
    pr_chunks = chunk_press_releases("data/rbi_docs/press_releases")
    all_chunks = booklet_chunks + pr_chunks
    print(f"total chunks: {len(all_chunks)}")
    for c in all_chunks[:3]:
        print("-", c["title"])

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("fraud_knowledge", embedding_function=ef)

    ids = [f"doc_{i}" for i in range(len(all_chunks))]
    documents = [c["text"] for c in all_chunks]
    metadatas = [{"title": c["title"], "source": c["source"]} for c in all_chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print("loaded into Chroma:", collection.count(), "chunks")