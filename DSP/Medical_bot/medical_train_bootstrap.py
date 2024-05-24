import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShot

from dspy.evaluate.metrics import answer_exact_match

from dspy.evaluate.evaluate import Evaluate
from dsp.utils import deduplicate

# Initialize the language model and retrieval model


lm = dspy.OllamaLocal(model='llama3:8b-instruct-q5_1',max_tokens=4000)
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








train_example = train_dataset[0]

train_sample =  train_dataset[:20]


input_key_only = train_example.inputs()
non_input_key_only = train_example.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)


# test_sample = dataset_test[:10]

print(train_example.keys())


#answering with basic uinput

class BasicQA(dspy.Signature):
    """Act as a Excellent Doctor and give solutions based on the symptops """

    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with links and medicenes if required  ")

generate_answer = dspy.ChainOfThought(BasicQA)



#retriver way
retrieve = dspy.Retrieve(k=3)

#now lets create a basic rag function 

class GenerateAnswer(dspy.Signature):
    """"Act as a Excellent Doctor and give solutions based on the symptops """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with links and medicenes if required")

class RAG(dspy.Module):
    def __init__(self, num_passages=4):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    

def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM



 # PERForming Multi hop search for data 
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
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
# Example usage


my_question = "I am facing joint paints after going to gym what to do  "   



# lm.inspect_history(n=3)


##uncomment this part to Train the model with some examples 
#Traininng
config = dict(max_bootstrapped_demos=5, max_labeled_demos=5)

teleprompter = BootstrapFewShot(metric=validate_context_and_answer,**config)
baleen = teleprompter.compile(SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=train_sample)

# # we are saving the model json here and to show how to run 
baleen.save('doctor.json')


model = SimplifiedBaleen()  # 




model.load('doctor.json')
pred = model(my_question)


# # Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

lm.inspect_history(n=1)