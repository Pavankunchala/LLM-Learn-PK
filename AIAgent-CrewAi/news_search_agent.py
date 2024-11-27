from crewai import Agent, LLM, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
import os
import datetime

# Set up environment variable for SerpAPI API key (replace "your_api_key_here" with your actual API key)
os.environ['SERPER_API_KEY'] = "your_api_key_here"

# Initialize LLM model with Ollama server running locally on port 11434
llm = LLM(
    model='ollama/mistral-nemo',
    base_url='http://localhost:11434'
)

# Initialize tools for news searching and scraping websites
tools = [
    SerperDevTool(),
    ScrapeWebsiteTool()
]

# Define the News Researcher agent with its role, goal, backstory, allowed delegation,
# verbosity settings, LLM model, and available tools.
news_agent = Agent(
    role="News Researcher",
    goal=(
        "Find the latest and most relevant news articles on the given {topic} "
        "at the given date {date}, and provide their headline, link, and a brief overview."
    ),
    backstory=(
        "You are a news searching agent who uses Google to find the latest news articles "
        "on various topics. Your goal is to retrieve the most accurate, relevant, and "
        "up-to-date information to support detailed reporting."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=tools
)

# Define the News Summarization agent with its role, goal, backstory, allowed delegation,
# verbosity settings, LLM model, and available tools.
news_summarization_agent = Agent(
    role="News Summarizer",
    goal=(
        "Take the news articles retrieved by the news agent and organize them into a detailed report. "
        "Each article should include: "
        "1. Headline (with link) "
        "2. Key Points (minimum 5 points summarizing important facts or takeaways) "
        "3. A detailed Summary (around 300-400 words) that provides comprehensive coverage of the article."
    ),
    backstory=(
        "You are a summarizing agent responsible for creating a detailed and structured report "
        "that highlights critical information and provides insights for decision-making."
    ),
    allow_delegation=False,
    llm=llm,
    verbose=True
)

# Define the searching task with its description, expected output format,
# and assigned news_agent.
searching_task = Task(
    description=(
        "Search for the top 10 most important and relevant news articles on the topic {topic} "
        "from Google News or other sources at the date {date}. Each result should include: "
        "1. Headline "
        "2. Link "
        "3. Brief overview of the article."
    ),
    expected_output=(
        "A list of 10 articles with their headline, link, and brief overviews, "
        "sorted by relevance and importance."
    ),
    agent=news_agent
)

# Define the summarizing task with its description, expected output format,
# output file location, and assigned news_summarization_agent.
summarizing_task = Task(
    description=(
        "Take the 10 articles retrieved by the searching task and produce a detailed report "
        "for each article in the format: "
        "1. Headline (with link) "
        "2. Key Points "
        "3. Summary. "
        "Each summary should include the critical details while maintaining conciseness."
    ),
    expected_output="A detailed Markdown report containing structured summaries for all 10 articles.",
    output_file='report.md',
    agent=news_summarization_agent
)

# Initialize the Crew with its agents and tasks, set verbosity to True.
crew = Crew(
    agents=[news_agent, news_summarization_agent],
    tasks=[searching_task, summarizing_task],
    verbose=True
)

# Define inputs for topic and date (current date is used as an example)
inputs = {
    "topic": "Drug Discovery with AI",
    "date": datetime.datetime.now().strftime("%Y-%m-%d")
}

# Kick off the tasks in the Crew using the provided inputs.
task_output = crew.kickoff(inputs=inputs)

# Print the task output (Markdown report).
print(task_output)