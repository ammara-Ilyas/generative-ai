import streamlit as st
from components.pdf_parser import get_pdf_text
from components.text_chunker import get_text_chunks
from components.vector_store_manager import get_vector_store
from components.retriever_loader import load_retriever
from components.chain_loader import load_conversational_chain

import uuid


def main():
    st.set_page_config("Chat with PDF", layout="wide")
    st.header("Chat with PDF using LangChain")

    # Initialize session-based conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = {}

    # Generate and store a unique session ID for the session
    session_id = st.session_state.get("session_id", str(uuid.uuid4()))
    st.session_state["session_id"] = session_id

    # Upload and process PDFs
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files", accept_multiple_files=True)
    if pdf_docs and st.sidebar.button("Submit & Process"):
        for pdf_doc in pdf_docs:
            with st.spinner(f"Processing {pdf_doc.name}..."):
                raw_text = get_pdf_text([pdf_doc])
                text_chunks = get_text_chunks(raw_text)
                # Save the vector store using PDF filename (without extension for consistency)
                get_vector_store(text_chunks, pdf_doc.name)
            st.success(f"Processing complete for {pdf_doc.name}")

    # Dropdown to select which PDF to query
    if pdf_docs:
        pdf_filenames = [pdf_doc.name for pdf_doc in pdf_docs]
        selected_pdf = st.sidebar.selectbox(
            "Select PDF to query", pdf_filenames)

        # Input area for asking questions
        user_question = st.text_input(
            "Ask a question from the selected PDF content:")

        # Check if the user inputs "bye" to end the chat
        if user_question.lower() == "bye":
            st.write("Chat session ended. Thank you!")
            # Store the current conversation history for this PDF in session state
            st.session_state.conversation_history[selected_pdf] = st.session_state.conversation_history.get(
                selected_pdf, []
            )
            st.stop()

        if user_question and user_question.lower() != "bye":
            # Load retriever for the selected PDF and initiate conversation
            try:
                retriever = load_retriever(selected_pdf)
                conversational_chain = load_conversational_chain(
                    retriever, session_id)

                # Query the chain with user question and selected PDF's conversation history
                response = conversational_chain({
                    "question": user_question,
                    "chat_history": st.session_state.conversation_history.get(selected_pdf, [])
                })

                # Display the bot's response
                st.write("Bot Reply:", response["answer"])

                # Update session-based conversation history for the selected PDF
                st.session_state.conversation_history.setdefault(selected_pdf, []).append(
                    {"question": user_question, "answer": response["answer"]}
                )

            except Exception as e:
                st.error(f"Error during retrieval: {e}")

    # Display all chat history by session in the right sidebar
    with st.sidebar:
        st.subheader("Chat Histories by PDF")
        if "conversation_history" in st.session_state:
            for pdf, history in st.session_state.conversation_history.items():
                st.write(f"Chat History for {pdf}:")
                for entry in history:
                    st.write(f"- {entry['question']}: {entry['answer']}")
