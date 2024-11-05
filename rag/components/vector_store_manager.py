from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os


def get_vector_store(text_chunks, pdf_filename):
    """Creates a FAISS vector store for embeddings and saves it by PDF filename."""
    print("Creating or loading FAISS vector store for embeddings.")

    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Set the FAISS index path based on the PDF filename (removing extension)
    faiss_index_path = f"data/{os.path.splitext(pdf_filename)[0]}.faiss"

    # Check if FAISS index already exists for the specific PDF
    if os.path.exists(f"{faiss_index_path}.index"):
        # Load existing FAISS index
        vector_store = FAISS.load_local(
            faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded existing FAISS index for {pdf_filename}.")
    else:
        # Create a new FAISS index if it does not exist
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        print(f"Created new FAISS index for {pdf_filename}.")

    # Save the FAISS index
    vector_store.save_local(faiss_index_path)
    print(f"FAISS index saved as {faiss_index_path}.")

    return vector_store
