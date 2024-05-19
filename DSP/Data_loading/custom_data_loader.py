import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate.metrics import answer_exact_match


#in this code we are loading a custom dataset from hugging face rather then ones that  are availbe in examples 

# Initialize the language model and retrieval model
lm = dspy.OllamaLocal(model='llama3:8b-instruct-q5_1',max_tokens=4000)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

# Load data
dl = DataLoader()
code_alpaca = dl.from_huggingface("HuggingFaceH4/CodeAlpaca_20K", split=["train", "test"])

#check the code aplaa train and see the keys
print("test",code_alpaca["train"][0])
dataset_train, dataset_test = [], []
for split in ['train', 'test']:
    for example in code_alpaca[split]:
        dsp_example = dspy.Example(question=example['prompt'], answer=example['completion']).with_inputs("prompt")
        if split == 'train':
            dataset_train.append(dsp_example)
        else:
            dataset_test.append(dsp_example)

train_example = dataset_train[0]

print(f"Question: {train_example.question}")
print(f"Answer: {train_example.answer}")

