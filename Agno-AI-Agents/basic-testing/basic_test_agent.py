from agno.agent import Agent
from agno.models.ollama import Ollama 

agent = Agent(
    model = Ollama(id = "gemma3:12b"),
    description = "You are an expert science concept explainer who will explain complex scienctif concepts in easy digestable chunks with examples",
    markdown = True
)


agent.print_response("Explain me the nature of light ", stream=True)