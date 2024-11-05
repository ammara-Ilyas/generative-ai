from langchain.memory import BaseChatMessageHistory, ChatMessageHistory
from langchain.memory.chat_message_history import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


class SessionBasedMemory:
    def __init__(self):
        # Dictionary to store chat histories for each session
        self.sessions = {}

    def get_history(self, session_id):
        """Retrieve chat history for a given session_id."""
        if session_id not in self.sessions:
            # Initialize a new history if session_id does not exist
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def add_message(self, session_id, role, content):
        """Add a message to a session's history."""
        history = self.get_history(session_id)
        history.add_message(role, content)


# Initialize Language Model, Vector Store, and Retriever as in previous example
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


documents = get_pdf_text("data/glucose.pdf")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Initialize session-based memory
session_memory = SessionBasedMemory()


def get_conversational_chain_for_session(session_id):
    # Retrieve or create chat history for the session
    chat_history = session_memory.get_history(session_id)

    # Use ConversationBufferMemory with chat history from the specific session
    memory = ConversationBufferMemory(
        input_key="question",
        memory_key="chat_history",
        output_key="answer",
        chat_memory=chat_history
    )

    # Initialize ConversationalRetrievalChain with session-based memory
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return conversational_chain


# Define a session ID
session_id = "session_123"

# Get or create a conversational chain for the specific session
conversational_chain = get_conversational_chain_for_session(session_id)

# Ask a question
response = conversational_chain({"question": "What is LangChain?"})
print("Response:", response["answer"])

# Retrieve and Print Chat History for the Specific Session
print("\nChat History for Session:", session_id)
chat_history = session_memory.get_history(session_id).messages
for message in chat_history:
    print(f"{message['role']}: {message['content']}")
