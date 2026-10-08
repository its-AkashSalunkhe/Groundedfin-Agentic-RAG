'''Next step: chunk and load into Chroma (talking about the BE(A)WARE booklet and RBI press releases).'''



import os, re, json
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

PART_C_HEADERS = [
    "General precautions",
    "For device / computer security",
    "For safe internet browsing",
    "For safe internet banking",
    "Factors indicating that a phone is being spied",
    "Actions to be taken after occurrence of a fraud",
    "Precautions related to Debit / Credit cards",
    "For E-mail account security",
    "For password security",
    "How do you know whether an NBFC accepting deposit is genuine or not?",
    "Precautions to be taken by depositors",
    "File a complaint",
]

def chunk_booklet(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    numbered_pattern = re.compile(r'\n\s*(\d{1,2})\.\s+([A-Z][^\n]+)')
    numbered_matches = list(numbered_pattern.finditer(full_text))
    boundaries = [(m.start(), m.group(2).strip()) for m in numbered_matches]

    # Part C headers aren't numbered, so we search for them by plain text --
    # but only AFTER the last numbered section, so we never accidentally match
    # an earlier mention of the same words sitting in the Table of Contents
    search_from = numbered_matches[-1].start() if numbered_matches else 0
    for header in PART_C_HEADERS:
        idx = full_text.find(header, search_from)
        if idx != -1:
            boundaries.append((idx, header))

    boundaries.sort(key=lambda b: b[0])

    chunks = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        body = full_text[start:end].strip()
        if len(body) > 80:
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
    booklet_path = r"data/rbi_docs/BEAWARE_booklet.pdf"
    pr_folder = r"data/rbi_docs/press_releases"
    
    booklet_chunks = chunk_booklet(booklet_path)
    pr_chunks = chunk_press_releases(pr_folder)
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