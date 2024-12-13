from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import BaseTool, ScrapeWebsiteTool
from crewai import Agent, Task ,LLM , Crew, Process
#llm
# Initialize LLM model with Ollama server running locally on port 11434
llm = LLM(
    model='ollama/llama3.2-vision',
    base_url='http://localhost:11434'
)


# DuckDuckGo Search Tool
class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Search the web for a given query."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response

# Instantiate Tools
search_tool = MyCustomDuckDuckGoTool()
scrape_tool = ScrapeWebsiteTool()


# Data Collector Agent
data_collector = Agent(
    role='Data Collector',
    goal=(
        "Gather detailed and relevant information on the research topic "
        "using DuckDuckGo and scraping tools."
    ),
    tools=[search_tool, scrape_tool],
    backstory=(
        "A highly skilled researcher specializing in finding, scraping, "
        "and curating reliable online resources."
    ),
    llm=llm
)

# Summarizer Agent
summarizer = Agent(
    role='Summarizer',
    goal=(
        "Condense the gathered data into actionable insights for each section "
        "of the research paper."
    ),
    backstory=(
        "An adept researcher with a knack for extracting key information "
        "from complex datasets."
    ),
    llm=llm
)

# Writer Agent
writer = Agent(
    role='Writer',
    goal=(
        "Draft a comprehensive and structured research paper adhering to IEEE standards, "
        "covering all essential sections."
    ),
    backstory=(
        "An experienced academic writer skilled at creating cohesive and well-structured "
        "papers from technical inputs."
    ),
    llm=llm
)

# Reviewer Agent
reviewer = Agent(
    role='Reviewer',
    goal=(
        "Review the research paper draft to ensure clarity, coherence, and adherence to IEEE standards."
    ),
    backstory=(
        "A meticulous academic reviewer with a strong eye for detail and adherence to formatting."
    ),
    llm=llm
)

tasks = [
    Task(
        description=(
            "Collect detailed and relevant information on {topic} using DuckDuckGo and ScrapeWebsiteTool. "
            "Identify at least 5 high-quality resources, including research papers, technical blogs, and whitepapers. "
            "For each resource, provide: \n"
            "- Title\n"
            "- Link\n"
            "- Summary (200-300 words per resource)"
        ),
        expected_output=(
            "A list of 5-7 detailed resources, including titles, links, and summaries of 200-300 words each."
        ),
        agent=data_collector
    ),
    Task(
        description=(
            "Summarize the collected information into detailed sections for the research paper. "
            "Specifically, extract insights for the following sections: Abstract, Introduction, "
            "Literature Review, Methodology, Future Work, and Conclusion. Ensure the output for each "
            "section provides substantial detail and clarity."
        ),
        expected_output=(
            "Summarized content for Abstract (~250 words), Introduction (~1 page), Literature Review (~2 pages), "
            "Methodology (~2 pages), Future Work (~1 page), and Conclusion (~1 page)."
        ),
        agent=summarizer
    ),
    Task(
        description=(
            "Draft a structured research paper adhering to IEEE standards, combining the outputs from the Summarizer Agent. "
            "Include the following sections: \n"
            "- Abstract (200-250 words)\n"
            "- Introduction (~1 page)\n"
            "- Literature Review (~2 pages)\n"
            "- Methodology (~2 pages)\n"
            "- Future Work (~1 page)\n"
            "- Conclusion (~1 page)\n"
            "- References (formatted in IEEE style)\n"
            "Ensure the paper is cohesive, well-structured, and clear."
        ),
        expected_output=(
            "A complete draft of the research paper (~10 pages) with all sections included and formatted in IEEE style."
        ),
        agent=writer
    ),
    Task(
        description=(
            "Review the research paper draft for clarity, coherence, grammar, and adherence to IEEE formatting. "
            "Ensure the paper is error-free, logically structured, and ready for submission."
        ),
        expected_output=(
            "A polished, finalized research paper (~10 pages) with all sections complete and formatted correctly."
        ),
        agent=reviewer,
        output_file = 'research_paper.md'
    )
]

crew = Crew(
    agents=[data_collector, summarizer, writer, reviewer],
    tasks=tasks,
    process=Process.sequential,
    verbose=True
)

#Kickoff the Workflow
inputs = {'topic': 'GPU Architecture and GPU Programming'}
result = crew.kickoff(inputs)

# Print Results
print("Research Workflow Result:", result)