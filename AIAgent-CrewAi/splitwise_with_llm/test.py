from vision_parse import VisionParser, PDFPageConfig
from crewai import Agent, Task, Crew, Process, LLM
llm = LLM(
    model='ollama/mistral-nemo',
    base_url='http://localhost:11434'
)

# Standalone function to convert PDF to Markdown
def convert_pdf_to_markdown(pdf_path: str) -> str:
    # Configure PDF page processing
    page_config = PDFPageConfig(
        dpi=400,
        color_space="RGB",
        include_annotations=False,
        preserve_transparency=True
    )

    # Initialize the VisionParser
    parser = VisionParser(
        model_name="llama3.2-vision:11b",
        temperature=0.2,
        top_p=0.3,
        extraction_complexity=False,
        page_config=page_config
    )

    # Process the PDF and return the Markdown content
    markdown_pages = parser.convert_pdf(pdf_path)
    result = "\n\n".join([f"--- Page {i+1} ---\n{content}" for i, content in enumerate(markdown_pages)])
    return result


# Define the agent
bill_splitter_agent = Agent(
    role="Bill Splitter",
    goal="Analyze the provided bill and determine the cost per person based on their purchases.",
    backstory=(
        "You are highly skilled at reading and interpreting complex billing documents. "
        "With an eye for detail, you ensure fairness by accurately breaking down bills "
        "into individual costs based on what each person bought. Everyone relies on you "
        "to resolve disputes over shared expenses."
    ),
    verbose=True,
    llm=llm
)

# Define the task
split_bill_task = Task(
    description=(
        "Analyze the provided bill in Markdown format and determine the cost per person "
        "based on the description of what each person bought.\n\n"
        "Inputs:\n"
        "- Bill Markdown Content: {markdown_content}\n"
        "- Purchases Description: {purchases}\n\n"
        "Example Purchases Description:\n"
        '"Alice bought drinks and appetizers. Bob bought the main course and dessert. '
        'Charlie bought dessert."'
    ),
    expected_output=(
        "A detailed breakdown of the total bill and individual costs based on purchases., give it in a neat table format, with detailed explain and clear explaination, for common items  divide it by 3 people like bread and other utensils , and also give each item in detailed in with added tax and how much it would cost if it divided among 3 people if its not mentioned in the bill , also for the remaining items add them among 3 people so  that the bill  tallies out it should all be in detailed markdown file   "
    ),
    agent=bill_splitter_agent,
    output_file='bill_output.md'
)

# Create the crew
crew = Crew(
    agents=[bill_splitter_agent],
    tasks=[split_bill_task],
    process=Process.sequential
)

# Example Usage
pdf_path = "bill.pdf"
markdown_content = convert_pdf_to_markdown(pdf_path)  # Convert the PDF to Markdown

# Inputs for the Crew
inputs = {
    "markdown_content": markdown_content,  # Use the generated Markdown content
    "purchases": {
        "pavan": ["protein bars", "milk"],
        "Bob": ["veg tofu"],
        "Charlie": ["protein powder" ,"turmeric gummy"]
    }
}

# Execute the Crew task
result = crew.kickoff(inputs=inputs)
print(result)
