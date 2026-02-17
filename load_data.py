import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import psycopg2
import uuid


load_dotenv()

genai_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=genai_api_key)

# print("List of models that support embedContent:\n")
# for m in client.models.list():
#     for action in m.supported_actions:
#         if action == "embedContent":
#             print(m.name)

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres", 
    user="postgres",              
    password=os.getenv("PG_PASSWORD") 
)
cur = conn.cursor()

def process_pdf(file_path, chunk_size=400, overlap=75):
    reader = PdfReader(file_path)
    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            all_chunks.append({
                "text": chunk_text,
                "page": page_number
            })

            start += (chunk_size - overlap)

    return all_chunks

def load_pdf(file_path):
    document_name = os.path.basename(file_path)
    chunks = process_pdf(file_path)

    for i, chunk in enumerate(chunks):
        embedding_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"]
        )

        vector = embedding_response.embeddings[0].values
        chunk_id = f"{document_name}_{i}_{uuid.uuid4().hex[:8]}"

        cur.execute("""
            INSERT INTO pdf_chunks
            (chunk_id, document_name, page_number, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            chunk_id,
            document_name,
            chunk["page"],
            chunk["text"],
            vector
        ))

    conn.commit()

    return f"Ingested {len(chunks)} chunks from {document_name}"

print(load_pdf("/Users/phanindravarma/Downloads/cobalt-hydrogen-bonded-organic-framework-as-a-visible-light-driven-photocatalyst-for-co2-cycloaddition-reaction.pdf"))