from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()


llm = GoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

try:
    loader = TextLoader("data.txt")
    print(loader.load())
except Exception as e:
    print("Error while loading text : ", e)
# print("loading")
# Create embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
index_creator = VectorstoreIndexCreator(
    embedding=embedding, text_splitter=text_splitter)

index = index_creator.from_loaders({loader})
# print("index", index)
while True:
    human_message = input("How i can help you today? ")
    response = index.query(human_message, llm=llm)
    print(response)
