# from crewai import Agent, Task, 
from crewai import Agent, LLM, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import BaseTool  
from crewai_tools import ScrapeWebsiteTool

search = DuckDuckGoSearchRun()

class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Search the web for a given query."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        
        response = duckduckgo_tool.invoke(query)

        return response
search_tool = MyCustomDuckDuckGoTool()
scrape_tool = ScrapeWebsiteTool()
llm = LLM(
    model='ollama/mistral-nemo',
    base_url='http://localhost:11434'
)



# Define Code Analyzer Agent
code_analyzer_agent = Agent(
    role="Code Analyzer",
    goal=(
        "Analyze the given code and explain each part, breaking it down into sections."
    ),
    backstory=(
        "You are a skilled software engineer with a deep understanding of programming concepts. "
        "Your primary job is to meticulously examine code, understand its functionality, and explain it in simple, clear terms."
    ),
    tools=[search_tool],
    llm=llm
)

# Define Content Formatter Agent
content_formatter_agent = Agent(
    role="Content Formatter",
    goal=(
        "Create a clear, well-formatted response from the analyzed explanation."
    ),
    backstory=(
        "You are a professional content creator with a talent for organizing technical information into engaging and visually appealing formats. "
        "Your job is to ensure the explanations are clear, structured, and accessible and really detailed  and you will use the search tool and scrape tool to gather more information to create more detailed and relevent blog"
    ),
    tools=[search_tool, scrape_tool],
    llm=llm
)

# Define Code Analysis Task
code_analysis_task = Task(
    description=(
        "Analyze the given code {code} and break it down into sections. Explain each part, starting with imports and moving sequentially. "
        "Highlight the purpose, functionality, and importance of each part. of the code explaining clearly everty functionality "
    ),
    expected_output=(
        "A raw, plain text explanation of the analyzed code, divided into clear sections., explaining \n"
        "A clear explaination of every part of code without miissing anythingh"
    ),
    
    agent=code_analyzer_agent,
     # Output key for piping to the next task
)

# Define Content Formatting Task
content_formatting_task = Task(
    description=(
        "Take the raw explanation from the analysis and format it into a polished response. The formatted response should:\n"
        "1. Include a title and headings for each section.\n"
        "2. explain each code block in detail don't miss anything from the main code .\n"
        "3. Add a detailed summary and say what other places can a similar architecutre can be used  "
    ),
    expected_output=(
        "A Markdown file containing the detailed explaination for every code block and in a detailed manner "
    ),
      # Piping the output from the first task
    agent=content_formatter_agent,
    context= [code_analysis_task],
    output_file="formatted_blog.md",
)


# Forming the Crew
crew = Crew(
    agents=[code_analyzer_agent, content_formatter_agent],
    tasks=[code_analysis_task, content_formatting_task],
    process=Process.sequential,  # Ensure sequential execution
    verbose=True,
)
code_input = """
 from crewai import Agent, LLM, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain.agents import Tool
from crewai_tools import BaseTool  

import os
import datetime


# Initialize LLM model with Ollama server running locally on port 11434
llm = LLM(
    model='ollama/llama3.2-vision',
    base_url='http://localhost:11434'
)
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Search the web for a given query."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        
        response = duckduckgo_tool.invoke(query)

        return response
search_tool = MyCustomDuckDuckGoTool()

# searching_tool = SerperDevTool()
scrape_tool  = ScrapeWebsiteTool()

room_finder_agent = Agent(
     role="Room Finder Agent",
    goal=(" Find a room to rent in the city or place you want to live {location}, within the price range of {price} per month on this date {date}."),
backstory=("You are a Room Finder Agent tasked with helping users locate rental accommodations in their desired location." "Your primary responsibility is to search for rooms or apartments available for rent that meet the user's specified criteria," "including location, budget, and other preferences." "You utilize advanced search tools and web scraping techniques to gather information from various online rental platforms," "classified ads, and real estate listings." "Your goal is to provide users with the most accurate, relevant, and up-to-date options" "that match their requirements." "Your summaries should include:"

"The name and description of the property."
"The monthly rental price."
"The location and proximity to landmarks or public transport."
"Any additional features or amenities provided, such as Wi-Fi, parking, or utilities included."
"Contact details or links to inquire about the property." "You aim to simplify the process of finding a rental by doing the heavy lifting of research and presenting the user with" "a curated list of options." "You are efficient, reliable, and user-focused, ensuring that the options you provide align closely" "with the user's needs and budget."),
     tools=[
         search_tool,
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




"""
# Execute the task
result = crew.kickoff(inputs={"code": code_input})
print(result)