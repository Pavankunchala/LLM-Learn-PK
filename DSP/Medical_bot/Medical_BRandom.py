import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
import re

# from dspy.evaluate.metrics import answer_exact_match

# from dspy.evaluate.evaluate import Evaluate
from dsp.utils import deduplicate

# import numexpr as ne

# ne.set_num_threads(28)

# Initialize the language model and retrieval model


lm = dspy.OllamaLocal(model='llama3',max_tokens=4000,model_type="text",timeout_s= 250)
metricLM = dspy.OllamaLocal(model='phi3', max_tokens=4000, model_type='text')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

# Load data

dl = DataLoader()
train_dataset = []
code_alpaca = dl.from_huggingface("lavita/ChatDoctor-HealthCareMagic-100k", split=["train"])
train_dataset, dataset_test = [], []
for split in ['train']:
    for example in code_alpaca[split]:
        dsp_example = dspy.Example(question=example['input'], answer=example['output']).with_inputs("question")

        if split == 'train':
            train_dataset.append(dsp_example)
    

# print("test",code_alpaca["train"][0])

# article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

# input_key_only = article_summary.inputs()
# non_input_key_only = article_summary.labels()

# print("Example object with Input fields only:", input_key_only)
# print("Example object with Non-Input fields only:", non_input_key_only)








train_example = train_dataset[40]

train_sample =  train_dataset[:10]
test_sample = train_dataset[20:40]


input_key_only = train_example.inputs()
non_input_key_only = train_example.labels()

# print("Example object with Input fields only:", input_key_only)
# print("Example object with Non-Input fields only:", non_input_key_only)


# # test_sample = dataset_test[:10]

print(train_example.keys())


#answering with basic uinput

class BasicQA(dspy.Signature):
    """Act as a Excellent Doctor and give solutions based on the symptops """

    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with links and medicenes if required  ")

generate_answer = dspy.ChainOfThought(BasicQA)



#retriver way
# retrieve = dspy.Retrieve(k=3)

#now lets create a basic rag function 

class GenerateAnswer(dspy.Signature):
    """Act as a Amazing Doctor with broad medical knowledge, and give detailed Diagnostic the with links and remedy if required  based on the  symptoms and problems given by the User """

    context = dspy.InputField(desc="Helpful information for answering the question.")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Give Detailed Answers to problem presented by the user with Links and Medication if required based on the diagnostic act as DOCTOR")

class RAG(dspy.Module):
    def __init__(self, num_passages=4):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    




 # PERForming Multi hop search for data 
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    


class Assess(dspy.Signature):
    """Assess the quality of an answer to a question, as Medical Evaluator and give a rating for outputs from 1.0 to 5.0 """
    
    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer : float = dspy.OutputField(desc="A float rating between 1.0 and 5.0 . IMPORTANT!! ONLY OUTPUT THE RATING!!",ge=1.0, le=5.0)


def extract_answer(text):
    # Improved pattern to capture numbers following specific keywords, handling both integers and floats
    patterns = [
        r'Rating:\s*(\d+(?:\.\d+)?)',     # Capture 'Rating: <number>'
        r'Review:\s*(\d+(?:\.\d+)?)',     # Capture 'Review: <number>'
        r'Feedback Score:\s*(\d+(?:\.\d+)?)', # Capture 'Feedback Score: <number>'
        r'Performance:\s*(\d+(?:\.\d+)?)',  # Capture 'Performance: <number>'
        r'Quality:\s*(\d+(?:\.\d+)?)',      # Capture 'Quality: <number>'
        r'Mark:\s*(\d+(?:\.\d+)?)',         # Capture 'Mark: <number>'
        r'Points:\s*(\d+(?:\.\d+)?)',       # Capture 'Points: <number>'
        r'Total:\s*(\d+(?:\.\d+)?)',        # Capture 'Total: <number>'
        r'Evaluation Score:\s*(\d+(?:\.\d+)?)', # Capture 'Evaluation Score: <number>'
        r'Result:\s*(\d+(?:\.\d+)?)',       # Capture 'Result: <number>'
        r'Assessment Answer:\s*(\d+(?:\.\d+)?)', # Capture 'Assessment Answer: <number>'
        r'Detail:\s*(\d+(?:\.\d+)?)',       # Capture 'Detail: <number>'
        r'Faithful:\s*(\d+(?:\.\d+)?)',     # Capture 'Faithful: <number>'
        r'Score:\s*(\d+(?:\.\d+)?)',        # Capture 'Score: <number>'
        r'rating of\s*(\d+(?:\.\d+)?)'      # Capture 'rating of <number>'
    ]

    # Compile all patterns into a single regular expression
    combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)

    # Search for all matches in the text
    matches = combined_pattern.findall(text)

    # Filter out empty matches and convert strings to float
    extracted_values = [float(num) for nums in matches for num in nums if num]

    # Check if any values were extracted and return the first one
    if extracted_values:
        return extracted_values[0]
    else:
        return 0  # or c
def llm_metric(gold, pred, trace=None):
    predicted_answer = pred.answer
    question = gold.question
    
    # print(f"Test Question: {question}")
    # print(f"Predicted Answer: {predicted_answer}")
    
    detail = "Is the assessed answer detailed? ,If so give a rating from 1.0 to 5.0"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context.,,If so give a rating from 1.0 to 5.0"
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`,If so give a rating from 1.0 to 5.0"
    
    with dspy.context(lm=metricLM):
        context  = dspy.Retrieve(k=3)(question).passages
        detail = dspy.ChainOfThought(Assess)(context="N/A", assessed_question=detail, assessed_answer=predicted_answer)
        faithful = dspy.ChainOfThought(Assess)(context=context, assessed_question=faithful, assessed_answer=predicted_answer)
        overall = dspy.ChainOfThought(Assess)(context=context, assessed_question=overall, assessed_answer=predicted_answer)

    detail_score =extract_answer( detail.assessment_answer)
    faithful_score = extract_answer(faithful.assessment_answer)
    overall_score = extract_answer(overall.assessment_answer)
    


   
    print(f"Faithful: {faithful.assessment_answer} - Score: {faithful_score}")
    print(f"Detail: {detail.assessment_answer} - Score: {detail_score}")
    print(f"Overall: {overall.assessment_answer} - Score: {overall_score}")    
    
    total = float(detail_score) + float(faithful_score)*2 + float(overall_score)

    

    total = total/25
    print(f"Total: {total}")
    
    return total
# Example usage
class TypedEvaluator(dspy.Signature):
    """Evaluate the quality of a system's answer to a question according to a given criterion."""
    
    criterion: str = dspy.InputField(desc="The evaluation criterion.")
    question: str = dspy.InputField(desc="Rewrite the question in the way you like for the input ")
    ground_truth_answer: str = dspy.InputField(desc="Given ground truth answer from other AI model ")
    predicted_answer: str = dspy.InputField(desc="The system's answer to the question. ")
    rating: float = dspy.OutputField(desc="A float rating between 1 and 5. IMPORTANT!! ONLY OUTPUT THE RATING!!", ge=0., le=1)


def MetricWrapper(gold, pred, trace=None):
    alignment_criterion = "How aligned is the predicted_answer with the ground_truth answer given by other AI model?"
    return dspy.TypedPredictor(TypedEvaluator)(criterion=alignment_criterion,
                                          question=gold.question,
                                          ground_truth_answer=gold.answer,
                                          predicted_answer=pred.answer).rating

    

    


my_question = """ I am facing Back Pain after playing basket Ball  act as a Doctor and give diagnostics and remedies with Medicences """


# lm.inspect_history(n=3)


##uncomment this part to Train the model with some examples 
# # Traininng
# config = dict(max_bootstrapped_demos=1,num_threads = 28, max_labaled_demos=2)

# teleprompter = BootstrapFewShotWithRandomSearch(metric=llm_metric,max_bootstrapped_demos=2,max_labeled_demos=2,num_candidate_programs=4,num_threads=28)
# baleen = teleprompter.compile(RAG(),trainset=train_sample,valset= test_sample)

# # # # we are saving the model json here and to show how to run 
# baleen.save('doctor_Brandom_Llama.json')


model = RAG()  # 




model.load('doctor_Brandom.json')
pred = model(my_question)


# # Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
# print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

# lm.inspect_history(n=1)