from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
import os
load_dotenv()
#      llm

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
#    prebuilt tools
tools = load_tools(["calculator"])
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHORT_REACT_DESCRIPTION,
    verbose=True
)
# agent
agent = initialize_agent(
    tools,  # The tools it can use
    llm,    # The language model (ChatOpenAI here)
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent type
    verbose=True  # Prints out what's happening step-by-step
)

result = agent.run("What is 15 plus 10?")
print(result)  # It will use the calculator tool to give you the answer!
