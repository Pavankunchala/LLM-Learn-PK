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
code_alpaca = dl.from_huggingface("HuggingFaceH4/CodeAlpaca_20K", split=["train", "test"])
dataset_train, dataset_test = [], []
for split in ['train', 'test']:
    for example in code_alpaca[split]:
        dsp_example = dspy.Example(question=example['prompt'], answer=example['completion']).with_inputs("question")
        if split == 'train':
            dataset_train.append(dsp_example)
        else:
            dataset_test.append(dsp_example)

train_example = dataset_train[0]

train_sample =  dataset_train[:10]

test_sample = dataset_test[:10]

# print(f"Question: {train_example.question}")
# print(f"Answer: {train_example.answer}")

#answering with basic uinput

class BasicQA(dspy.Signature):
    """Answer questions with detailed answers and code ."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with code if required ")

generate_answer = dspy.ChainOfThought(BasicQA)



#retriver way
retrieve = dspy.Retrieve(k=3)

#now lets create a basic rag function 

class GenerateAnswer(dspy.Signature):
    """Answer questions with detailed answers and code ."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with code if required")

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


#creating a baleen model which will get some answers and  creates some more queries from them and gets answers and consolidates it 
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
    

my_question = "How do neural networks work, give me some real life comparisons  "



# lm.inspect_history(n=3)


##uncomment this part to Train the model with some examples 
#Traininng
# config = dict(max_bootstrapped_demos=5, max_labeled_demos=5)

# teleprompter = BootstrapFewShot(metric=validate_context_and_answer,**config)
# baleen = teleprompter.compile(SimplifiedBaleen(), trainset=train_sample)

# we are saving the model json here and to show how to run 
# baleen.save('multihop.json')


model = SimplifiedBaleen()  # 




model.load('multihop.json')
pred = model(my_question)


# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

lm.inspect_history(n=1)