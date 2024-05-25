import dspy 

from dspy import (
    InputField,
    OutputField,
    Predict,
    ChainOfThought,
    Signature
)
from rich import print


lm = dspy.OllamaLocal(model='llama3:8b-instruct-q5_1',max_tokens=4000)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

# Load data

#lets define a custom class 

#Lets assume we have a dataset with some paitent details where we are trying to classify some data according to their name, age, gender, disease 


class PaitentClassifier(Signature):
    """Classify the given Paitent data into their Name, Age, Gender,Disease """

    sentence = InputField(desc = "data to be classified")
    data_type = OutputField(desc = "Categorized data based on the above classes ")



paitent_pred = Predict(PaitentClassifier)


# will give the input and output fields required for the classifer and also kind of see how it looks bit under the hood 
print(paitent_pred)

#lets also  create one model that generates all the input as well or if not bothering let me create some input data from llms 

input_sentence = "In a recent conversation, I learned about Emma, a young girl of 8 struggling with asthma, Mr. Clark who at 65 has just been diagnosed with diabetes, and Janet, a 42-year-old battling breast cancer."

test = paitent_pred(sentence = input_sentence)

print(test.data_type)

""" this is the output given by the classifier:

Here is the classified data:

**Name**: Emma, Mr. Clark, Janet
**Age**: 8, 65, 42
**Gender**: Female (Emma), Male (Mr. Clark)
**Disease**: Asthma, Diabetes, Breast Cancer

"""