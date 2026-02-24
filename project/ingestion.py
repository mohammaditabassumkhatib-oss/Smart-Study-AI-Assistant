import pdfplumber
import re
import chromadb
import os
from embeddings import embed_documents
from config import CHROMA_DIR

def secondary_chunking(text, max_chars=3000):
    """
    Break down large text blocks into smaller segments to avoid API token limits.
    """
    sub_chunks = []
    while len(text) > max_chars:
        # Try to split at the last period within the limit to keep sentences whole
        split_index = text.rfind(". ", 0, max_chars)
        if split_index == -1:
            split_index = max_chars
        else:
            split_index += 1 # Include the period
            
        sub_chunks.append(text[:split_index].strip())
        text = text[split_index:].strip()
    
    if text:
        sub_chunks.append(text)
    return sub_chunks

def load_and_chunk_pdf(pdf_path: str) -> list[str]:
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    # 1. Split by Section Headers (e.g., 1.1, 2.3)
    section_pattern = r"\n\d+\.\d+.*"
    sections = re.split(section_pattern, all_text)
    
    # 2. Filtering and Cleaning
    chunks = [s.strip() for s in sections if len(s.strip()) > 100]

    # 3. Logic to Merge Examples/Continuations (Your existing logic)
    merged = []
    for section in chunks:
        is_example = any(k in section.lower() for k in ["example", "illustration", "table"])
        if is_example and merged:
            merged[-1] += "\n\n" + section
        else:
            merged.append(section)

    # 4. SAFETY CHECK: Final Token-Limit Splitter
    final_safe_chunks = []
    for chunk in merged:
        if len(chunk) > 4000: # Rough character limit to stay safe under 8k tokens
            final_safe_chunks.extend(secondary_chunking(chunk))
        else:
            final_safe_chunks.append(chunk)

    print(f"Total safe chunks created: {len(final_safe_chunks)}")
    return final_safe_chunks

def build_vector_store(chunks):
    # Use PersistentClient to ensure the 'chroma' folder is created and populated
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    collection = client.get_or_create_collection("ml_notes")

    # Generate Embeddings
    embeddings = embed_documents(chunks)
    ids = [str(i) for i in range(len(chunks))]

    # Add to Chroma
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    
    print(f"Success! Vector store built at: {CHROMA_DIR}")