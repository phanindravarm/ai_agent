import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import psycopg2
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

def process_pdf(file_path):
    reader = PdfReader(file_path)
    all_chunks = []
  
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 300,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for page_number, page in enumerate(reader.pages, start = 1):
        text = page.extract_text()

        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "page": page_number
        })
            
    return all_chunks

def load_pdf(file_path):
    document_name = os.path.basename(file_path)

    cur.execute("""
        INSERT INTO documents (document_name) VALUES (%s)
        RETURNING document_id
    """, (document_name,))

    document_id = cur.fetchone()[0]

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
            (chunk_id, document_id, page_number, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            chunk_id,
            document_id,
            chunk["page"],
            chunk["text"],
            vector
        ))

    conn.commit()

    return f"Ingested {len(chunks)} chunks from {document_name}"

print(load_pdf("/Users/phanindravarma/Downloads/TIMETABLE CHANGES 2 II SEM 2025 -26.pdf"))