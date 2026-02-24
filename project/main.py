import os
import sys
from ingestion import load_and_chunk_pdf, build_vector_store
from retrieval import retrieve_top_k
from generation import generate_response
from config import CHROMA_DIR

def main():
    print("\n--- Smart Study AI Assistant ---\n")
    
    # 1. Check if the Vector Store exists
    if not os.path.exists(CHROMA_DIR):
        pdf_file = "ml_notes.pdf"
        if not os.path.exists(pdf_file):
            print(f"Error: {pdf_file} not found in project directory!")
            return
        
        print(f"Initializing system: Processing {pdf_file}...")
        try:
            chunks = load_and_chunk_pdf(pdf_file)
            build_vector_store(chunks)
            print("Vector store created successfully.")
        except Exception as e:
            print(f"An error occurred during ingestion: {e}")
            return
    else:
        print("Knowledge base loaded.")

    # 2. Chat Loop
    while True:

        print("\nChoose mode:")
        print("1. Summary")
        print("2. QA")
        print("3. Quiz")
        print("Type 'exit' to quit")

        choice = input("Enter choice (1/2/3): ").strip()

        if choice.lower() in ["exit", "quit", "q"]:
            print("Goodbye! Happy studying.")
            break

        valid_choices = {
            "1": "summary",
            "summary": "summary",
            "2": "qa",
            "qa": "qa",
            "3": "quiz",
            "quiz": "quiz"
        }

        if choice.lower() not in valid_choices:
            print("Invalid choice. Try again.")
            continue

        mode = valid_choices[choice.lower()]

        if choice == "1":
            mode = "summary"
            user_query = input("Enter topic to summarize: ").strip()

        elif choice == "2":
            mode = "qa"
            user_query = input("Enter your question: ").strip()

        elif choice == "3":
            mode = "quiz"
            user_query = input("Enter topic for quiz: ").strip()

        if not user_query:
            continue

        print("AI is thinking...")

        try:
            context = retrieve_top_k(user_query, mode = mode)
            answer = generate_response(mode, context, user_query)
            print(f"\nAI Assistant:\n{answer}")

        except Exception as e:
            print(f"Oops! Something went wrong: {e}")

if __name__ == "__main__":
    main()