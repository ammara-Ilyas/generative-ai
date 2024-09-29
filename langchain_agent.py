from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import tool
import re
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM with GoogleGenerativeAI
llm = GoogleGenerativeAI(model="gemini-1.5-flash",
                         google_api_key=os.getenv("GOOGLE_API_KEY"))


@tool
def add_numbers_tool(input_data: str) -> str:
    """Addition of numbers from a given input string."""
    # Extract all numbers from the string
    numbers = re.findall(r'\d+', input_data)
    if len(numbers) < 2:
        return "Please provide at least two numbers for addition."
    int_numbers = list(map(int, numbers))
    result = sum(int_numbers)
    return f"The sum of numbers {",".join(map(str, int_numbers))} is {result}"


agent = initialize_agent(
    tools=[add_numbers_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    max_iterations=1,
)

# Test the agent with a valid input
output = agent.run(
    "I went to the shop and bought mangoes for 100, bananas for 50, apples for 200, and oranges for 500."
)
print(output)
