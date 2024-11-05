from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import python_multipart
import os
import uvicorn

load_dotenv()
app = FastAPI()

# Google API key setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up LLM using Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Define a function to extract text from PDF files


def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create or load a FAISS vector store with Google embeddings


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    else:
        vector_store = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store


# A dictionary to store session histories
session_histories = {}


@app.post("/process-pdf/")
async def process_pdf(files: list[UploadFile]):
    raw_text = get_pdf_text(files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return {"status": "PDFs processed successfully"}


@app.post("/ask-question/")
async def ask_question(user_question: str, session_id: str):
    # Load FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    # Initialize conversational chain
    qa_prompt = ChatPromptTemplate.from_template("{input}")
    conversational_rag_chain = ConversationalRetrievalChain(
        llm=llm,
        retriever=retriever,
        combine_docs_chain=qa_prompt,
        return_source_documents=True
    )

    # Retrieve session-specific chat history
    chat_history = session_histories.get(session_id, [])

    # Run the retrieval-augmented generation chain
    response = conversational_rag_chain.invoke({
        "input": user_question,
        "chat_history": chat_history
    })

    # Extract answer and source documents
    answer = response["answer"]
    source_docs = response.get("source_documents", [])

    # Update session history with the new question and answer
    chat_history.append({"user": user_question, "assistant": answer})
    session_histories[session_id] = chat_history
    print("history", chat_history)
    # Return the generated answer
    return {"answer": answer, "sources": [doc["text"] for doc in source_docs]}

# Uncomment to run FastAPI app
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
