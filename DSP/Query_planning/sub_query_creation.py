import dspy 




lm = dspy.OllamaLocal(model="mistral-nemo",max_tokens=10000,temperature=0.2)

dspy.settings.configure(lm = lm)

 
class subquery(dspy.Signature):
    """ Act as a Sub query generator , for the given question , generate 5 sub questions , that divides the complex question into smaller and more easily answerable parts
    """
    question = dspy.InputField(desc = "Input questions")
    sub_queries = dspy.OutputField(desc= "5 sub questions for the given question , making the intial question more easily aanswerable")



test = dspy.ChainOfThought(subquery)

divided_questions  = test(question = "what is dark matter , how do we find it")

print(divided_questions.sub_queries)