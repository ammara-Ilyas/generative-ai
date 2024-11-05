from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

text_chunks = ""


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        print("pdf", pdf)
        pdf_reader = PdfReader(pdf)
        print("pages length", len(pdf_reader.pages))
        for page in pdf_reader.pages:
            text += page.extract_text()
    # print("text", text)
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    print("chunk", chunks)
    print("chunk", len(chunks))
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("data/faiss")
    print("FAISS vector store created and saved.")
    return vector_store


# def load_vector_store():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     # Load the saved FAISS index from the local path
#     vector_store = FAISS.load_local(
#         "data/faiss", embeddings, allow_dangerous_deserialization=True)
#     print("FAISS vector store loaded.")
#     return vector_store


# def query_vector_store(vector_store, query_text, top_k=3):
#     # Convert the query text to an embedding
#     results = vector_store.similarity_search(query_text, k=top_k)
#     # return results
#     # Print out the results
#     for idx, result in enumerate(results):
#         print(f"Result {idx+1}:")
#         print(f"Content: {result.page_content}")
#         print(f"Metadata: {result.metadata}")
#         print()

# Load the vector store


def load_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the FAISS vector store
    vector_store = FAISS.load_local(
        "data/faiss", embeddings, allow_dangerous_deserialization=True)
    # Create a retriever from the FAISS vector store
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})
    print("Retriever loaded.")
    return retriever


def query_retriever(retriever, query_text):
    # Query the retriever with the input text
    results = retriever.get_relevant_documents(query_text)
    return results
    # Print the results
    # for idx, result in enumerate(results):
    #     print(f"Result {idx+1}:")
    #     print(f"Content: {result.page_content}")
    #     print(f"Metadata: {result.metadata}")
    #     print()


def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using GPT-4💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process") and pdf_docs:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.success("Processing complete. You can now ask questions.")

    if user_question:
        # vector_store = load_vector_store()
        # results = query_vector_store(vector_store, user_question, 3)
        # print("response", results)
        retriever = load_retriever()
        results = query_retriever(retriever, user_question)
        # results = query_vector_store(vector_store, user_question, 3)

        # user_input(user_question, retriever)
        for idx, result in enumerate(results):
            print(f"Result {idx+1}:")
            print(f"Content: {result.page_content}")
            print(f"Metadata: {result.metadata}")
            print()

            st.write("Reply: ", results.result.page_content)

        # retriever = load_retriever()
        # query_retriever(retriever, query_text)

        # user_input(user_question, retriever)


if __name__ == "__main__":
    main()
