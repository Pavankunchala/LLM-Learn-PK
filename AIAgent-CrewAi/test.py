
from crewai.cli.constants import ENV_VARS
ENV_VARS.update(
    {"ollama":
      [{
            "prompt": "Enter your OLLAMA API_BASE (press Enter to skip)",
            "key_name": "API_BASE",
        },
        # {
        #     "default": True,
        #     "API_BASE": "http://localhost:11434",
        # }
    ]})

from crewai import Agent , LLM, Task, Crew


llm = LLM(
    model = 'ollama/llama3.2-vision',
    base_url='http://localhost:11434'
)
 

blog_writing_agent = Agent(

    role = "Blog Writing Expert",
    goal = "craft high-impact, engaging blogs that captivate audiences and convey complex ideas in an accessible manner",
    backstory="""  You're a seasoned wordsmith with a knack for distilling complex information into compelling stories. With years of experience writing on diverse topics, you've honed your ability to craft clear, concise, and informative content that resonates with readers. \
        Your expertise spans multiple industries, and you're adept at adapting your tone and style to suit any audience or topic. Whether it's explaining cutting-edge AI concepts or sharing insights from the latest machine learning research, you have a talent for making complex ideas accessible to all.""",
    llm = llm ,
    verbsoe = True

    
)


blog_task = Task(
    description= "Find and summarize the latest happening most relevant news on AI",
    agent=blog_writing_agent,
     expected_output='A bullet list summary of the top 5 most important AI news',
)

# Execute the crew
crew = Crew(
    agents=[blog_writing_agent],
    tasks=[blog_task],
    verbose=True
)


result = crew.kickoff()

# Accessing the task output
task_output = blog_task.output

print(f"Task Description: {task_output.description}")
print(f"Task Summary: {task_output.summary}")
print(f"Raw Output: {task_output.raw}")