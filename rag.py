from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_text_splitters import CharacterTextSplitter
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
text_splitter = CharacterTextSplitter(
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

index_creator = VectorstoreIndexCreator(
    embedding=embeddings, text_splitter=text_splitter)

index = index_creator.from_loaders({loader})
print("hello5")
# print("index", index)
while True:
    human_message = input("How i can help you today?")
    response = index.query(human_message, llm=llm)
    print(response)
