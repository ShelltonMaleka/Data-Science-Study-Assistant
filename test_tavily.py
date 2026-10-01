import os

from dotenv import load_dotenv
from tavily import TavilyClient


# Load the API key from the .env file
load_dotenv()

# Get the Tavily API key from the environment
api_key = os.getenv("TAVILY_API_KEY")

# Make sure the API key exists before starting the search
if not api_key:
    raise ValueError(
        "TAVILY_API_KEY was not found. "
        "Check that your .env file contains the API key."
    )


# Create the Tavily search client
client = TavilyClient(
    api_key=api_key
)


# Search the web for a test question
query = "What is a convolutional neural network?"

response = client.search(
    query=query,
    search_depth="basic",
    max_results=5
)


# Display the search results
print("\nTavily search results:\n")

for result in response["results"]:

    print("Title:", result.get("title"))
    print("URL:", result.get("url"))
    print("Content:", result.get("content"))
    print("-" * 80)