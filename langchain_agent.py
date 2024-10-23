from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import Tool, initialize_agent, AgentType
import re
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM with GoogleGenerativeAI
llm = GoogleGenerativeAI(model="gemini-1.5-flash",
                         google_api_key=os.getenv("GOOGLE_API_KEY"))


@tool
# (name="add Numbers", description="If user says to add or something which show that addition process then add it.Multiplies numbers from the given input string.")
def add_numbers_tool(input_data: str) -> str:
    """Addition of numbers from a given input string."""
    # Extract all numbers from the string
    numbers = re.findall(r'\d+', input_data)
    if len(numbers) < 2:
        return "Please provide at least two numbers for addition."
    int_numbers = list(map(int, numbers))
    result = sum(int_numbers)
    return f"The sum of numbers {",".join(map(str, int_numbers))} is {result}"


@tool
# (name="Multiply Numbers", description="If user says to multiply or something which show that multiplication process then multiply it.Multiplies numbers from the given input string.")
def multiply_numbers_tool(input_data: str) -> str:
    """Addition of numbers from a given input string."""
    # Extract all numbers from the string
    numbers = re.findall(r'\d+', input_data)
    if len(numbers) < 2:
        return "Please provide at least two numbers for addition."
    int_numbers = list(map(int, numbers))
    result = 1

    # Multiply all numbers together
    for num in int_numbers:
        result *= num

    # Return the result with a formatted message
    return f"The product of numbers {', '.join(map(str, int_numbers))} is {result}"


# Initialize the tools using Tool.from_function
tools = [
    Tool.from_function(
        func=add_numbers_tool,
        name="Add Numbers",
        description="Adds numbers from the given input string."
    ),
    Tool.from_function(
        func=multiply_numbers_tool,
        name="Multiply Numbers",
        description="Multiplies numbers from the given input string."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=1,
)

# Test the agent with a valid input
output = agent.run(
    "I went to the shop and bought mangoes for 100, bananas for 50 and give 500  to every sister and I have 3 sisters and tell me to calculate it"
)
print(output)
