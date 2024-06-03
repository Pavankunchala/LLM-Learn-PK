import dspy

from dspy import (
    Signature,
    InputField,
    OutputField,
    Module,

)
lm = dspy.OllamaLocal(model='qwen:0.5b', max_tokens=4000, model_type="text", timeout_s=250)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)



from pydantic import BaseModel, Field

class Input(BaseModel):
    context: str = Field(description="The context for the question")
    query: str = Field(description="The question to be answered")

class Output(BaseModel):
    answer: str = Field(description="The answer for the question")
    confidence: float = Field(ge=0, le=1, description="The confidence score for the answer")


class QASignature(dspy.Signature):
    """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are about the answer."""

    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()


predictor = dspy.TypedChainOfThought(QASignature)

doc_query_pair = Input(
    context="The quick brown fox jumps over the lazy dog",
    query="What does the fox jumps over?",
)

prediction = predictor(input=doc_query_pair)

answer = prediction.output.answer
confidence_score = prediction.output.confidence

print(f"Prediction: {prediction}\n\n")
print(f"Answer: {answer}, Answer Type: {type(answer)}")
print(f"Confidence Score: {confidence_score}, Confidence Score Type: {type(confidence_score)}")