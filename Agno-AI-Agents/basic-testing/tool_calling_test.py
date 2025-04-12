from agno.agent import Agent
from agno.models.ollama import Ollama 
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model = Ollama(id = "llama3.2"),
    description = "You are an expert science concept explainer who will explain complex scienctif concepts in easy digestable chunks with examples",
    tools = [ DuckDuckGoTools()],
    markdown = True
)


agent.print_response("Explain me the latest Double slit in time experiment how does it work and what does it prove and who researched about it  ", stream=True)