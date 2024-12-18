import os
import pandas as pd
import matplotlib.pyplot as plt
from crewai import Crew, Task, Agent, LLM
from crewai_tools import FileReadTool, CSVSearchTool
import chardet


# Set the API Key for LLM
os.environ['GROQ_API_KEY'] = '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'

# Detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# File path to dataset
dataset_path = './customer_support_tickets.csv'

# Detect encoding dynamically
file_encoding = detect_encoding(dataset_path)
print(f"Detected encoding: {file_encoding}")

# Attempt to load the dataset
try:
    df = pd.read_csv(dataset_path, encoding=file_encoding, )
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Fallback to a forgiving encoding
    df = pd.read_csv(dataset_path)
    print("Dataset loaded with fallback encoding!")

# Preprocessing to handle missing or problematic data
df.fillna("Unknown", inplace=True)
print("Data after preprocessing:")
print(df.head())

# Save cleaned data to a new file (optional)
df.to_csv('./cleaned_customer_support_ticket_dataset.csv', index=False)

# Initialize LLM model
llm = LLM(
    model='ollama/mistral-nemo',
    base_url='http://localhost:11434',
)

# Tool to load and process dataset
csv_tool = CSVSearchTool(file_path='./cleaned_customer_support_ticket_dataset.csv', config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="mistral-nemo",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="ollama", # or openai, ollama, ...
            config=dict(
                model="nomic-embed-text",
                
                # title="Embeddings",
            ),
        ),
    ))

# Agents
data_ingestion_agent = Agent(
    role="Data Ingestion Specialist",
    goal="Efficiently load and preprocess the customer support ticket data for analysis.",
    backstory="You handle large datasets, ensuring they are clean and ready for analysis.",
    tools=[csv_tool],
    llm=llm,
    verbose=True
)

issue_classification_agent = Agent(
    role="Issue Classification Expert",
    goal="Categorize support tickets into predefined issue types to facilitate targeted analysis.",
    backstory="Specializes in identifying patterns and categorizing issues to streamline support processes.",
    tools=[csv_tool],
    llm=llm,
    verbose=True
)

resolution_analysis_agent = Agent(
    role="Resolution Analyst",
    goal="Analyze the resolution times and statuses to identify bottlenecks and areas for improvement.",
    backstory="Analytical skills enable you to dissect resolution metrics, enhancing operational efficiency.",
    tools=[csv_tool],
    llm=llm,
    verbose=True
)

customer_satisfaction_agent = Agent(
    role="Customer Satisfaction Specialist",
    goal="Assess customer feedback and satisfaction ratings to evaluate the effectiveness of support resolutions.",
    backstory="Passionate about customer experience, ensuring high satisfaction levels are maintained.",
    tools=[csv_tool],
    llm=llm,
    verbose=True
)

report_generation_agent = Agent(
    role="Report Generation Expert",
    goal="Compile findings into a comprehensive report for stakeholders.",
    backstory="Transforms data and insights into professionally formatted reports.",
    tools=[csv_tool],
    llm=llm,
    verbose=True
)

# Tasks
data_ingestion_task = Task(
    description="Load the cleaned CSV file containing customer support tickets and perform any necessary preprocessing steps.",
    expected_output="A cleaned and structured dataset ready for analysis.",
    agent=data_ingestion_agent
)

issue_classification_task = Task(
    description="Analyze and categorize support tickets into issue types, e.g., Technical Issue, Billing Issue.",
    expected_output="Dataset with an additional column indicating issue category for each ticket.",
    agent=issue_classification_agent
)

resolution_analysis_task = Task(
    description="""
    Evaluate resolution times and statuses to identify trends such as:
    - Average resolution time by issue type.
    - Proportion of unresolved tickets.
    """,
    expected_output="Summary report detailing resolution metrics and identified bottlenecks.",
    agent=resolution_analysis_agent
)

customer_satisfaction_task = Task(
    description="""
    Analyze customer feedback and satisfaction ratings:
    - Identify trends in satisfaction scores.
    - Correlate resolution metrics with satisfaction levels.
    """,
    expected_output="Summary report on customer satisfaction with actionable recommendations.",
    agent=customer_satisfaction_agent
)

final_report_task = Task(
    description="""
    Integrate findings into a detailed report with:
    - Issue distribution charts and trends.
    - Resolution time analysis.
    - Customer satisfaction insights.
    - Actionable recommendations and visualizations.
    """,
    expected_output="A comprehensive report saved as 'final_report.md'.",
    agent=report_generation_agent,
    context=[
        data_ingestion_task,
        issue_classification_task,
        resolution_analysis_task,
        customer_satisfaction_task
    ],
    output_file='final_report.md'
)

# Crew
support_analysis_crew = Crew(
    agents=[
        data_ingestion_agent,
        issue_classification_agent,
        resolution_analysis_agent,
        customer_satisfaction_agent,
        report_generation_agent
    ],
    tasks=[
        data_ingestion_task,
        issue_classification_task,
        resolution_analysis_task,
        customer_satisfaction_task,
        final_report_task
    ],
    verbose=True
)

# Kickoff the crew
result = support_analysis_crew.kickoff()

# Save and display the result
print("Crew Execution Complete. Final report generated.")
print(result)
