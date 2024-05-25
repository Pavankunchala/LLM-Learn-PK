import dspy

from dspy import(
    Predict,
    Example,
    Signature,
    settings,
    InputField,
    OutputField,
    ChainOfThought

)

from rich import print 


# We are trying to understand more about how we define Examples for training the data in DSPY

test_example= Example(question = "What is the name of the tallest mountain?", answer = "Mount Everest", instruction = "Search about the tallest  mountain")


#this is the vanialla example 
print(test_example)

#now lets define the input key for the above example 


test_example_with_input  = test_example.with_inputs("question")

print("with input",test_example_with_input)

# and anything which is not the input will be the label

print("Labels",test_example_with_input.labels())


# for making it more as in a list 

dataset = [Example(doc = "long doc",summary = "summary of the document ").with_inputs("doc")]

print("with input in array",dataset[0].inputs())



# Lets look into  a basic metric to judge our answers right
class Asses(Signature):
    """Asses the quality of the output and give a rating between 1 to 5 """

    given_answer = InputField()
    given_question = InputField()
    rating = OutputField(desc = "Rating Between 1 to 5 ")
