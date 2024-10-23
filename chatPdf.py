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
    # Using Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create embeddings from text chunks
    embedded_texts = embeddings.embed_documents(text_chunks)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
   # Save the FAISS index to disk using FAISS's native method
    vector_store.save_local("data/faiss.txt")
    print("FAISS vector store created and saved.")


def get_conversational_chain():
    prompt_template = """
    Answer the question using the given context. If the answer is not in the context, say "answer is not available in the context".\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    # You can use any model like GPT-4
    # model = ChatGoogleGenerativeAI(model="gpt-4", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    # Load the chain with the LLM and the prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=None, prompt=prompt)

    return chain

# 3. Process user input and answer the question


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the stored Chroma DB

    # Load the stored FAISS index from disk using FAISS's native method
    vector_store = FAISS.load_local(
        "data/faiss.txt", embeddings, allow_dangerous_deserialization=True)

    # Use retriever to get relevant documents
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(user_question)
    # Get the conversational chain
    chain = get_conversational_chain()

    # Get the response from the chain
    response = chain.run({"context": docs, "question": user_question})

    print(response)
    st.write("Reply: ", response)

# 4. Main Streamlit App


def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using GPT-4💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)  # Store the chunks in Chroma
                st.success("Done")


if __name__ == "__main__":
    main()
