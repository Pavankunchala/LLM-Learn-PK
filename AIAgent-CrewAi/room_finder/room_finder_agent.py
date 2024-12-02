from crewai import Agent, LLM, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

import os
import datetime

# Set up environment variable for SerpAPI API key (replace "your_api_key_here" with your actual API key)
os.environ['SERPER_API_KEY'] = "your_api_here"

# Initialize LLM model with Ollama server running locally on port 11434
llm = LLM(
    model='ollama/llama3.2-vision',
    base_url='http://localhost:11434'
)


searching_tool = SerperDevTool()
scrape_tool  = ScrapeWebsiteTool()

room_finder_agent = Agent(
     role="Room Finder Agent",
    goal=(""" Find a room to rent in the city or place you want to live {location}, within the price range of {price} per month on this date {date}.
"""),
backstory=("""
You are a Room Finder Agent tasked with helping users locate rental accommodations in their desired location. 
Your primary responsibility is to search for rooms or apartments available for rent that meet the user's specified criteria, 
including location, budget, and other preferences.

You utilize advanced search tools and web scraping techniques to gather information from various online rental platforms, 
classified ads, and real estate listings. Your goal is to provide users with the most accurate, relevant, and up-to-date options 
that match their requirements. 

Your summaries should include:
1. The name and description of the property.
2. The monthly rental price.
3. The location and proximity to landmarks or public transport.
4. Any additional features or amenities provided, such as Wi-Fi, parking, or utilities included.
5. Contact details or links to inquire about the property.

You aim to simplify the process of finding a rental by doing the heavy lifting of research and presenting the user with 
a curated list of options. You are efficient, reliable, and user-focused, ensuring that the options you provide align closely 
with the user's needs and budget.
"""),
     tools=[
          searching_tool,
          scrape_tool
      ],



     llm = llm,
     
)




room_finding_task = Task(
    description=(
        "Find rental accommodations in the specified location {location} within the price range of {price} per month. "
        "Search across multiple online platforms, classified ads, and real estate listings to gather the most relevant options. "
        "The results should include detailed information about each rental property that matches the user's criteria."
    ),
    expected_output=(
        "A list of rental options in Markdown format with the following details for each property: "
        "1. Property name and description. "
        "2. Monthly rental price. "
        "3. Location (including proximity to landmarks or public transportation). "
        "4. Amenities (e.g., Wi-Fi, parking, utilities included). "
        "5. Contact details or links to inquire about the property."
    ),
    agent=room_finder_agent,
   
      output_file="room_output.md"
)



inputs = {
    "location": "Melbourne , FL",
     
    "price": "2500",
    "date": datetime.datetime.now().strftime("%Y-%m-%d")
}

crew = Crew(
    agents=[room_finder_agent],
    tasks=[room_finding_task],
    verbose=True

)
# Kick off the tasks in the Crew using the provided inputs.
task_output = crew.kickoff(inputs=inputs)

# Print the task output (Markdown report).
print(task_output)



