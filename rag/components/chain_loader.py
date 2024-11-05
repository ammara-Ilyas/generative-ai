from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import GOOGLE_API_KEY


def load_conversational_chain(retriever, session_id):
    """Creates a conversational retrieval chain with session-based memory."""
    print("Creating conversational retrieval chain with session-based memory.")

    # Set up the Language Model using Google GenAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )

    # Initialize memory that uses session_id to maintain conversation context
    memory = ConversationBufferMemory(
        input_key="question",
        memory_key="chat_history",
        output_key="answer",
        session_id=session_id  # Enable session-based memory
    )

    # Create a conversational chain using the LLM, retriever, and memory
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return conversational_chain
