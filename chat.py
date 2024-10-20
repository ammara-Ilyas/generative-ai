from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()
app = FastAPI()

# Google API key setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store = FAISS.load_local(
#     "faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    else:
        vector_store = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True)
        index = index_creator.from_loaders({loader})

    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    say "Answer is not available in the context."\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


@app.post("/process-pdf/")
async def process_pdf(files: list[UploadFile]):
    raw_text = get_pdf_text(files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return {"status": "PDFs processed successfully"}


@app.post("/ask-question/")
async def ask_question(user_question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the FAISS index
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search to retrieve relevant documents
    docs = vector_store.similarity_search(user_question)

    # Initialize your LLM (Google Generative AI in this case)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Load the QA chain with the LLM
    chain = load_qa_chain(llm, chain_type="stuff")

    # Use the chain to generate an answer based on the retrieved documents and user question
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Return the generated answer
    return {"answer": response["output_text"]}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
