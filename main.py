from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt = PromptTemplate(
    template="Write story of two friend who buy fruits? use these characters name {character}, write story like dilogue start with character name and after name should be : like Abc:, and every character name start from new line and name first name capital in urdu ", inputVriable=["character"])
chain = prompt | llm
# response = chain.invoke({"character": "Ali and Hamza"})
while True:
    human_message = input("Write character's name : ")
    ai_message = chain.invoke({"character": human_message})
    print(human_message)
    print(ai_message)
