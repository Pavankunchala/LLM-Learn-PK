import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShot

from dspy.evaluate.metrics import answer_exact_match


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

train_sample =  dataset_test[:10]

# print(f"Question: {train_example.question}")
# print(f"Answer: {train_example.answer}")

#answering with basic uinput

class BasicQA(dspy.Signature):
    """Answer questions with detailed answers and code ."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with code if required ")

generate_answer = dspy.ChainOfThought(BasicQA)

# pred = generate_answer(question=train_example.question)

# # Print the input and the prediction.
# print(f"Question: {train_example.question}")
# print(f"Thought: {pred.rationale.split('.', 1)[1].strip()}")

# print(f"Predicted Answer: {pred.answer}")


# lm.inspect_history(n=1)

#retriver way
retrieve = dspy.Retrieve(k=3)
# topK_passages = retrieve("how to resize image in python").passages

# print(f"Top {retrieve.k} passages for question: {train_example.question} \n", '-' * 30, '\n')

# for idx, passage in enumerate(topK_passages):
#     print(f'{idx+1}]', passage, '\n')


# ans = retrieve("When was the first FIFA World Cup held?").passages[0]
# print(ans)

#now lets create a basic rag function 

class GenerateAnswer(dspy.Signature):
    """Answer questions with detailed answers and code ."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with code if required")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
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

# training 

config = dict(max_bootstrapped_demos=3, max_labeled_demos=3)

teleprompter = BootstrapFewShot(metric=validate_context_and_answer,**config)
compiled_rag = teleprompter.compile(RAG(), trainset=train_sample)


compiled_rag.save("custom1.json")

# loading the model 

model = RAG()

model.load('custom1.json')

answer = model("How to resize an image in C++ give me code ")

print(answer)
