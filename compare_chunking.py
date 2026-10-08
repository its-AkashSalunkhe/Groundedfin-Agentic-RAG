import csv, re
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_compare")


def chunk_by_section(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    pattern = re.compile(r'\n\s*(\d{1,2})\.\s+([A-Z][^\n]+)')
    matches = list(pattern.finditer(full_text))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        if len(body) > 80:
            chunks.append(body)
    return chunks


def chunk_fixed_size(pdf_path, size=500, overlap=100):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    chunks = []
    start = 0
    while start < len(full_text):
        chunk = full_text[start:start + size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def build_collection(name, chunks):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.get_or_create_collection(name, embedding_function=ef)
    col.add(ids=[f"c{i}" for i in range(len(chunks))], documents=chunks)
    return col


PDF_PATH = "data/rbi_docs/BEAWARE_booklet.pdf"

section_chunks = chunk_by_section(PDF_PATH)
fixed_chunks = chunk_fixed_size(PDF_PATH)
print(f"section-based: {len(section_chunks)} chunks")
print(f"fixed-size: {len(fixed_chunks)} chunks")

section_col = build_collection("compare_section", section_chunks)
fixed_col = build_collection("compare_fixed", fixed_chunks)

with open("test_set.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["category"] == "STATIC"]

results = []
for row in rows:
    q = row["question"]
    s_result = section_col.query(query_texts=[q], n_results=1)
    f_result = fixed_col.query(query_texts=[q], n_results=1)
    s_dist = s_result["distances"][0][0]
    f_dist = f_result["distances"][0][0]
    winner = "section" if s_dist < f_dist else "fixed"
    results.append((row["id"], q, s_dist, f_dist, winner))
    print(f"{row['id']:4} | section={s_dist:.3f} | fixed={f_dist:.3f} | winner={winner} | {q}")

avg_section = sum(r[2] for r in results) / len(results)
avg_fixed = sum(r[3] for r in results) / len(results)
section_wins = sum(1 for r in results if r[4] == "section")

print(f"\nAverage distance -- section-based: {avg_section:.3f} | fixed-size: {avg_fixed:.3f}")
print(f"Section-based won {section_wins}/{len(results)} questions (lower distance = better match)")

with open("chunking_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "question", "section_distance", "fixed_distance", "winner"])
    writer.writerows(results)
print("Saved to chunking_comparison.csv")