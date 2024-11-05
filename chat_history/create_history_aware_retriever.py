import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
#      llm

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
#    load text from txt and pdf files
try:
    loader = PyPDFLoader(
        "./data/chem.pdf",
    )
    data = loader.load()
    # print(data)
    print("hello2")
except FileNotFoundError as e:
    print("File not found : ", e)
except Exception as e:
    print("Error while loading text : ", e)

#   split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
texts = text_splitter.split_documents(data)
# print(texts[0])
print("hello4")
dat = llm.invoke("What is your name?")
print(dat)
#  embedded
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create embeddings from text chunks
embedded_texts = embeddings.embed_documents(texts)
vector_store = FAISS.from_documents(texts, embedded_texts)
# vector_store = FAISS.from_documents(split_dos, embeddings)

# Save the FAISS index to disk using FAISS's native method
vector_store.save_local("data/faiss.txt")
retriever = vector_store.as_retriever()

print("FAISS vector store created and saved.")
print("hello5")
# print("index", index)


### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Step 8: Dictionary to Store Session Histories
session_histories = {}

# Step 9: Define a Function to Ask Questions and Store History by `session_id`


def ask_question(session_id, question):
    # Retrieve the chat history for this session, or create an empty one if it doesn't exist
    chat_history = session_histories.get(session_id, [])

    # Run the retrieval chain with question and chat history
    response = conversational_rag_chain(
        {"input": question, "chat_history": chat_history})

    # Extract answer and source documents
    answer = response["answer"]
    source_docs = response.get("source_documents", [])

    # Print the answer and source documents for reference
    print(f"Session {session_id} - Assistant:", answer)
    if source_docs:
        print("Source Documents:", [doc["text"] for doc in source_docs])

    # Update the session history with the new question and answer
    chat_history.append({"user": question, "assistant": answer})

    # Save updated history back into session_histories dictionary
    session_histories[session_id] = chat_history


# Example usage: Managing multiple sessions with different session IDs
session_id_1 = "session_1"
session_id_2 = "session_2"

# Ask questions in session 1
ask_question(session_id_1, "What is LangChain?")
ask_question(session_id_1, "Does it support RAG?")

# Ask questions in session 2
ask_question(session_id_2, "Can you explain LangChain's capabilities?")
ask_question(session_id_2, "What features does it offer?")

# Check the stored history for each session
print("\nSession Histories:")
for session, history in session_histories.items():
    print(f"{session}:", history)


# conversational_rag_chain.invoke(
#     {"input": "What is Task Decomposition?"},
#     config={
#         "configurable": {"session_id": "abc123"}
#     },  # constructs a key "abc123" in `store`.
# )["answer"]

# # second
# conversational_rag_chain.invoke(
#     {"input": "What are common ways of doing it?"},
#     config={"configurable": {"session_id": "abc123"}},
# )["answer"]
