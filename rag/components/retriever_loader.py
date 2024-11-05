from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_retriever():
    """Loads the saved FAISS vector store and sets it up as a retriever."""
    print("Loading FAISS vector store and setting up as a retriever.")

    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the existing FAISS index from local storage
    vector_store = FAISS.load_local(
        "data/faiss", embeddings, allow_dangerous_deserialization=True)

    # Return as a retriever with similarity search settings
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
