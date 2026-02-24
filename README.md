# Smart Study AI Assistant 

Smart Study AI Assistant is an AI-powered study tool designed to help students learn efficiently from their own study material (notes, textbooks, syllabi).

The system uses **Retrieval-Augmented Generation (RAG)** to ensure answers are grounded in the uploaded content, reducing hallucinations and improving reliability.

---

## Features

- Upload PDF notes or textbooks
- Convert documents into semantic embeddings
- Store embeddings in a **persistent Chroma vector database**
- Retrieve the most relevant content using **top-k vector similarity search**
- Answer student queries using retrieved context
- Designed to support:
  - Concept explanations
  - Notes generation
  - Question answering
  - (Upcoming) MCQs & summaries

---

## System Architecture (RAG Pipeline)

1. **Document Ingestion**
   - Load PDF
   - Clean and chunk text
   - Merge example/illustration sections when needed

2. **Embedding Generation**
   - Convert text chunks into dense vectors using OpenAI embeddings

3. **Vector Storage**
   - Store embeddings in a persistent Chroma collection

4. **Retrieval**
   - Embed user query
   - Retrieve top-k most relevant chunks using vector similarity

5. **Generation**
   - Send retrieved context to the LLM
   - Generate grounded answers based only on the source material

---

## Tech Stack

- **Python**
- **OpenAI API** (Embeddings + LLMs)
- **ChromaDB** (Persistent Vector Database)
- **Cosine Similarity / Vector Search**
- **Jupyter Notebook** (for experimentation & prototyping)

---

## 📂 Project Structure
