#######################
# Imports & Dependencies
#######################
from langchain_community.tools import WikipediaQueryRun  # pip install wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool  # pip install youtube_search
from langchain_community.tools.openai_dalle_image_generation import (
   OpenAIDALLEImageGenerationTool
)
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

import os

#######################
# Environment Setup
#######################
os.environ["OPENAI_API_KEY"] = "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"

#######################
# Tool Initialization
#######################
# Wikipedia Tool Setup
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(description="A tool to explain things in text format. Use this tool if you think the user’s asked concept is best explained through text.", api_wrapper=wiki_api_wrapper)
# print(wikipedia.invoke("Mobius strip"))

# DALL-E Tool Setup
dalle_api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1792x1024")
dalle = OpenAIDALLEImageGenerationTool(
   api_wrapper=dalle_api_wrapper, description="A tool to generate images. Use this tool if you think the user’s asked concept is best explained through an image."
)
# output = dalle.invoke("A mountain bike illustration.")
# print(output)

# YouTube Tool Setup
youtube = YouTubeSearchTool(
   description="A tool to search YouTube videos. Use this tool if you think the user’s asked concept can be best explained by watching a video."
)
# youtube.run("Oiling a bike's chain")

# Combine all tools
tools = [wikipedia, dalle, youtube]

#######################
# Model Configuration
#######################
chat_model = ChatOpenAI()
model_with_tools = chat_model.bind_tools(tools)

#######################
# Agent Execution Logic
#######################
#Testing

# response = model_with_tools.invoke([
#    HumanMessage("Can you generate an image of a mountain bike?")
# ])
# print(f"Text response: {response.content}")
# print(f"Tools used in the response: {response.tool_calls }")


# Creating a simple agent
def execute(agent, query, thread_id="a1b2c3"):
    """
    Execute the agent with a given query and thread ID.
    
    Args:
        agent: The configured agent instance
        query: User's input query
        thread_id: Unique identifier for the conversation thread
    
    Returns:
        Response from the agent
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        response = agent.invoke({'messages': [HumanMessage(query)]}, config=config)
        for message in response["messages"]:
            print(
                f"{message.__class__.__name__}: {message.content}"
            )  # Print message class name and its content
            print("-" * 20, end="\n")
        return response
    except Exception as e:
        print(f"Error executing agent: {e}")
        raise

#######################
# Agent Configuration
#######################
# System prompt definition
system_prompt = SystemMessage(
   """
   You are a helpful bot named Chandler. Your task is to explain topics
   asked by the user via three mediums: text, image or video.
  
   If the asked topic is best explained in text format, use the Wikipedia tool.
   If the topic is best explained by showing a picture of it, generate an image
   of the topic using Dall-E image generator and print the image URL.
   Finally, if video is the best medium to explain the topic, conduct a YouTube search on it
   and return found video links.
   """
)

# Agent creation with error handling
try:
    agent = create_react_agent(
        model=chat_model,  # Changed back to model from llm
        tools=tools,
        state_modifier=system_prompt
    )
except Exception as e:
    print(f"Error creating agent: {e}")
    raise

#######################
# Test Execution
#######################
# response = execute(agent, query='Explain the Fourier Series visually.')

# Let’s test it again:
response = execute(
   agent, query="what is blockchain", thread_id="123"
)