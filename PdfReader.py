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


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("data/faiss")
    print("FAISS vector store created and saved.")
    return vector_store


# def get_conversational_chain(retriever):
#     prompt_template = """
#     Answer the question using the given context. If the answer is not in the context, say "answer is not available in the context".\n\n
#     Context:\n {context}\n
#     Question: \n{question}\n
#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template,
#                             input_variables=["context", "question"])
#     chain = RetrievalQA.from_chain_type(
#         llm=llm, chain_type="stuff", retriever=retriever, prompt=prompt)
#     return chain


def user_input(user_question, retriever):
    response = user_input(user_question, retriever)

    st.write("Reply: ", response)


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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("data/faiss", embeddings)
        retriever = vector_store.as_retriever()
        rsponse = user_input(user_question, retriever)
        print("")
        st.write("Reply: ", rsponse.content)


if __name__ == "__main__":
    main()
