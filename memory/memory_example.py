# 1. ConversationBufferMemory
# This stores all interactions throughout the conversation.


from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.memory import CombinedMemory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationStringBufferMemory
from langchain.memory import ConversationBufferMemory

# Initialize memory
memory = ConversationBufferMemory()

# Simulate conversation
memory.save_context({"input": "Hello!"}, {"output": "Hi! How can I help you?"})
memory.save_context({"input": "What's the weather like?"},
                    {"output": "It's sunny."})

# Retrieve conversation history
print(memory.load_memory_variables({})["history"])
# Expected Output:

# Hello!
# Hi! How can I help you?
# What's the weather like?
# It's sunny.


# Advantages:
# Full Context: Retains the entire conversation, making it ideal for conversations where previous context is always relevant.
# Disadvantages:
# Memory Intensive: As the conversation grows, memory consumption increases, which could affect performance in long conversations.

# 2. ConversationStringBufferMemory
# This also stores all interactions, but in a single string format, which can be useful for certain LLMs that prefer text input.


# Initialize memory
memory = ConversationStringBufferMemory()

# Simulate conversation
memory.save_context({"input": "Tell me a joke"}, {
                    "output": "Why don't scientists trust atoms? Because they make up everything!"})
memory.save_context({"input": "Another one, please"}, {
                    "output": "What do you call fake spaghetti? An impasta!"})

# Retrieve conversation history
print(memory.load_memory_variables({})["history"])


# Tell me a joke
# Why don't scientists trust atoms? Because they make up everything!
# Another one, please
# What do you call fake spaghetti? An impasta!


# Advantages:
# Single String: Easier to integrate with some models that expect a single string of text as input.
# Disadvantages:
# Formatting Limitation: May not be ideal if different formats or structures in conversation history are needed.


# 3. ConversationBufferWindowMemory
# Only keeps the latest messages within a defined window size, which limits memory usage while keeping recent context.


# Initialize memory with a window size of 2 (keeps only the last 2 exchanges)
memory = ConversationBufferWindowMemory(k=2)

# Simulate conversation
memory.save_context({"input": "What's the capital of France?"}, {
                    "output": "Paris"})
memory.save_context({"input": "What's the population of Paris?"}, {
                    "output": "2.16 million"})
memory.save_context({"input": "Can you tell me about Eiffel Tower?"}, {
                    "output": "It's a famous landmark in Paris."})

# Retrieve conversation history
print(memory.load_memory_variables({})["history"])

# What's the population of Paris?
# 2.16 million
# Can you tell me about Eiffel Tower?
# It's a famous landmark in Paris.


# Advantages:
# Efficient Memory Use: Limits memory usage by only keeping recent context.
# Ideal for Recent Context: Suitable for scenarios where only the latest interactions are relevant.
# Disadvantages:
# Lost Historical Context: Older interactions are removed, which may limit understanding if earlier parts of the conversation are referenced.


# 4. CombinedMemory
# Combines multiple memory types, allowing customized memory handling, like storing short-term memory with a window and keeping a summary of long-term memory.


# Initialize CombinedMemory with buffer for entire conversation and window for recent exchanges
combined_memory = CombinedMemory(memories=[
    ConversationBufferMemory(),
    ConversationBufferWindowMemory(k=2)
])

# Simulate conversation
combined_memory.save_context({"input": "Hi there!"}, {
                             "output": "Hello! How can I assist you today?"})
combined_memory.save_context({"input": "Tell me a fact about Mars"}, {
                             "output": "Mars is known as the Red Planet."})
combined_memory.save_context({"input": "What about its atmosphere?"}, {
                             "output": "It has a thin atmosphere primarily composed of CO2."})

# Retrieve all memories
print("All conversation history:",
      combined_memory.load_memory_variables({})["memory_0"]["history"])
print("Recent history:", combined_memory.load_memory_variables(
    {})["memory_1"]["history"])


# All conversation history:
# Hi there!
# Hello! How can I assist you today?
# Tell me a fact about Mars
# Mars is known as the Red Planet.
# What about its atmosphere?
# It has a thin atmosphere primarily composed of CO2.

# Recent history:
# Tell me a fact about Mars
# Mars is known as the Red Planet.
# What about its atmosphere?
# It has a thin atmosphere primarily composed of CO2.


# Advantages:
# Flexible and Comprehensive: Combines multiple memory strategies, ideal for applications requiring both long-term and recent memory.
# Disadvantages:
# Complexity: Adds complexity in managing and retrieving from multiple memory sources.


# Summarization-Based:

# Retains essential information, avoiding the buildup of large amounts of conversation data.
# Efficiency: Helps manage memory usage in long interactions by summarizing rather than retaining every message.


# Initialize the summary memory with a language model (LLM) for summarization
llm = ChatOpenAI()  # Assume OpenAI LLM is initialized here
memory = ConversationSummaryMemory(llm=llm)

# Simulate conversation
memory.save_context({"input": "Can you tell me about the solar system?"}, {
                    "output": "Sure! The solar system consists of the Sun and the objects orbiting it, including eight planets."})
memory.save_context({"input": "Tell me about Mars specifically."}, {
                    "output": "Mars is the fourth planet from the Sun, often called the Red Planet."})
memory.save_context({"input": "Is there water on Mars?"}, {
                    "output": "Yes, there is evidence of frozen water on Mars, especially at its poles."})

# Retrieve the summarized conversation history
print(memory.load_memory_variables({})["history"])

# Summary:
# - User asked about the solar system, and learned it includes the Sun and planets.
# - User asked specifically about Mars, learning it's the fourth planet and called the Red Planet.
# - User inquired about water on Mars, finding out there’s evidence of frozen water, especially at the poles.
# Advantages:
# Concise Memory: Helps keep memory usage low by storing only a summary of key points.
# Scalable: Effective for longer conversations since it doesn’t retain every message but still provides the essential context.
# Improved Model Focus: Provides a focused summary that can improve the relevance of responses.
# Disadvantages:
# Loss of Details: Fine-grained details from the conversation are lost, which may not be suitable if precise recall is needed.
# Dependence on Summarization Quality: The quality of the summary depends on the summarization capability of the LLM used.
