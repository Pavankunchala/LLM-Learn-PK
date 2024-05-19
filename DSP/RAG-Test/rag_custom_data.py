import dspy
import dsp
from dspy.evaluate import answer_exact_match
from dspy.datasets import HotPotQA
from datasets import load_dataset
import numexpr as ne
from dspy.teleprompt import BootstrapFewShot

from dspy.datasets import DataLoader

from dspy import Prediction


class MyPrediction(Prediction):
    def __init__(self, prediction: Prediction):
        self.__dict__.update(vars(prediction))
        
    def __eq__(self, other):
        if isinstance(other, str):
            return False # because _store is intended to be a dict object
        return self._store == other._store

ne.set_num_threads(28)

#main settings for LLM and RM
lm = dspy.OllamaLocal(model='llama3:8b-instruct-q5_1',max_tokens=2400)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)


#loading the dataloader 
dl = DataLoader()


code_alpaca = dl.from_huggingface(
    "HuggingFaceH4/CodeAlpaca_20K",
    split = ["train", "test"],# returns Dictionary with train, test keys
     input_keys= ('prompt','completion') 
    
)

trainset = code_alpaca['train']
testset = code_alpaca['test']

# print(f"Keys present in the returned dict: {list(code_alpaca.keys())}")

# print(f"Number of examples in train set: {len(code_alpaca['train'])}")
# print(f"Number of examples in test set: {len(code_alpaca['test'])}")

# print(f"Number of examples in test set:{code_alpaca['train'][-1]}")


#just taking a few input values from thr dataset for testing 


trainset = code_alpaca['train'][:20]
devset = code_alpaca['test'][:10]
testset = code_alpaca['test'][10:30]


print(trainset[0])



trainset = [dspy.Example(question=question).with_inputs("prompt","completion") for question in trainset]
devset = [dspy.Example(question=question).with_inputs("prompt","completion") for question in devset]
testset = [dspy.Example(question=question).with_inputs("prompt","completion") for question in testset]


#lets also define a metric LLM


metricLM = dspy.OllamaLocal(model='llama3')


#and an asses function which would give a rating of 1-5
class Assess(dspy.Signature):
    """Assess the quality of an answer to a question."""
    
    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(desc="A rating between 1 and 5. Only output the rating and nothing else.")

def llm_metric(gold, pred, trace=None):
    predicted_answer = pred.completion
    question = gold.prompt
    
    print(f"Test Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")
    
    detail = "Is the assessed answer detailed?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`"
    
    with dspy.context(lm=metricLM):
        context = dspy.Retrieve(k=5)(question).passages
        detail = dspy.ChainOfThought(Assess)(context="N/A", assessed_question=detail, assessed_answer=predicted_answer)
        faithful = dspy.ChainOfThought(Assess)(context=context, assessed_question=faithful, assessed_answer=predicted_answer)
        overall = dspy.ChainOfThought(Assess)(context=context, assessed_question=overall, assessed_answer=predicted_answer)
    
    print(f"Faithful: {faithful.assessment_answer}")
    print(f"Detail: {detail.assessment_answer}")
    print(f"Overall: {overall.assessment_answer}")
    
    
    total = float(detail.assessment_answer) + float(faithful.assessment_answer)*2 + float(overall.assessment_answer)
    
    return total / 5.0




# #testing it on some randome questinms 
# test_example = dspy.Example(question="What do cross encoders do?")
# test_pred = dspy.Example(answer="They re-rank documents.")

# print(type(llm_metric(test_example, test_pred)))

# test_example = dspy.Example(question="What do cross encoders do?")
# test_pred = dspy.Example(answer="They index data.")

# print(type(llm_metric(test_example, test_pred)))



metricLM.inspect_history(n=3)


class GenerateAnswer(dspy.Signature):
    """Answer the prompts and in detailed and clear manner. if not use your own knowledge but answer correctly"""
    
   
    prompt = dspy.InputField()
    completion = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(answer=prediction.answer)
    

 #lets try it with predict    
    

uncompiled_rag= RAG()
# dspy.Predict(GenerateAnswer)(question="Write detailed code for resizing an image using C++")
# lm.inspect_history(n=1)


test_model = dspy.ReAct(GenerateAnswer)

#chain of thought
# dspy.ChainOfThought(GenerateAnswer)(question="Write detailed code for resizing an image using python")
# lm.inspect_history(n=1)
    


# dspy.ReAct(GenerateAnswer)(question="What are cross encoders?")
# lm.inspect_history(n=1)

from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(metric=llm_metric, max_labeled_demos=8, max_rounds=3)

# also common to init here, e.g. Rag()
compiled_rag = teleprompter.compile( test_model,trainset=trainset)

save_path = './v2.json'
compiled_rag.save(save_path)

compiled_rag("Write detailed code for resizing and change colors for image in python ").answer

lm.inspect_history(n=1)