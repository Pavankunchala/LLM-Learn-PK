import dspy
from dspy import ChainOfThought

# Initialize the LLM model for prompt refinement
lm = dspy.OllamaLocal(model="mistral-nemo", model_type="text", max_tokens=5000, temperature=0.0001, timeout_s=125)
dspy.settings.configure(lm=lm)

class PromptOptimizer(dspy.Signature):
    """
    You are an AI Assistant designed  designed to refine user queries for better retrieval from an LLM.
    Your task is to optimize the user input query by making it clearer, more specific, or more targeted
    toward retrieving the most relevant information.

    Requirements:
    - Each prompt should be refined to reduce ambiguity.
    - The prompt should be adjusted to guide the LLM toward better answers
    - Ensure that the refined prompt remains natural and user-friendly without adding any new or other information that will change the original question, make it really detailed and well formatted.
    """
    query = dspy.InputField(desc='The original user query to be optimized.')
    optimized_prompts = dspy.OutputField(desc='Only give the output prompts, nothing else, and make them as detailed as possible.')

def optimize_prompt(query: str, num_variations: int) -> list:
    """
    Generate refined versions of a user query for better retrieval from an LLM.
    
    Args:
        query (str): The original query from the user.
        num_variations (int): Number of optimized prompts to generate.
    
    Returns:
        list: A list of optimized prompt variations.
    """
    optimizer = ChainOfThought(PromptOptimizer)
    all_variations = []
    for _ in range(num_variations):
        variations = optimizer(query=query).optimized_prompts
        all_variations.append(variations)
    return all_variations

def save_to_txt(examples: list, output_file: str):
    """
    Save the optimized prompt examples to a txt file.
    
    Args:
        examples (list): List of generated prompt variations.
        output_file (str): The output file to save the examples.
    """
    with open(output_file, 'w') as file:
        for example in examples:
            file.write(f"{example}\n")

def generate_optimized_prompts(query: str, num_variations: int = 2) -> list:
    """
    This function is a wrapper that will return optimized prompts for a given query.
    
    Args:
        query (str): The original user query.
        num_variations (int): Number of prompt variations to generate (default is 2).
    
    Returns:
        list: The generated optimized prompts.
    """
    return optimize_prompt(query, num_variations)

# The following CLI setup is optional and can be used for command-line execution.
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize user prompts for better retrieval from an LLM.")
    parser.add_argument('--query', '-q', type=str, help="The user query to optimize.", required=True)
    parser.add_argument('--num_variations', '-n', type=int, default=2, help="The number of optimized prompts to generate.")
    parser.add_argument('--output', '-o', type=str, default='optimized_prompts.txt', help="The txt file to save the optimized prompts.")
    
    args = parser.parse_args()

    # Generate optimized prompts
    optimized_prompts = optimize_prompt(args.query, args.num_variations)

    # Save to output file
    save_to_txt(optimized_prompts, args.output)

    print(f"Generated {args.num_variations} optimized prompts for query '{args.query}' and saved to '{args.output}'.")

if __name__ == "__main__":
    main()
