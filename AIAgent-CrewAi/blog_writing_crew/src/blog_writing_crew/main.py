#!/usr/bin/env python
import sys
import warnings
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

from blog_writing_crew.crew import BlogWritingCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'Prompt engineering techniques for better responses from LLM'
    }
    BlogWritingCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "Prompt engineering techniques for better responses from LLMPrompt engineering techniques for better responses from LLM"
    }
    try:
        BlogWritingCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        BlogWritingCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "Prompt engineering techniques for better responses from LLMPrompt engineering techniques for better responses from LLM"
    }
    try:
        BlogWritingCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
