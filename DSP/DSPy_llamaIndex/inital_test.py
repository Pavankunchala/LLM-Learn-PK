#the goal here is to use llama index and dspy together to have a retrival and prompt engineering model 

from dspy import (
    Signature,
    InputField,
    OutputField,
    Predict,
    ChainOfThought
)
import openai
import dspy
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage

from llama_index.core.embeddings import resolve_embed_model
from langchain_community.llms import Ollama
from rich import print



lm = dspy.OllamaLocal(model='llama3:8b-instruct-q5_1',max_tokens=10000,model_type="text",timeout_s= 125)
Settings.llm =  Ollama(model="llama3:8b-instruct-q5_1")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)
#lets first load the documents

document_directory = "data"

documents = SimpleDirectoryReader(document_directory).load_data()

Settings.embed_model = resolve_embed_model("local:thenlper/gte-small")

#lets start indexing the documents
index = VectorStoreIndex.from_documents(documents,show_progress=True)

index.set_index_id("vector_index")
index.storage_context.persist("./storage")

storage_context = StorageContext.from_defaults(persist_dir="storage")

# Create query engine as index
index = load_index_from_storage(storage_context, index_id="vector_index")
query_engine = index.as_query_engine(response_mode="refine")

# Create signature
class GenerateAnswer(Signature):
    """Answer questions with detailed answers as Research Analyst, using retrieved context for accuracy."""
    context = InputField(desc="May contain relevant facts from retrieved documents")
    question = InputField(desc="The question that needs an answer")
    answer = OutputField(desc="Give detailed answers and more information about the answer and relevant links if necessary")
#
#lets create a basic rag funcrtion 

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.query_engine = query_engine
        self.generate_answer = ChainOfThought(GenerateAnswer)
        print("Class 2 created")

    def forward(self, question):
        response = self.query_engine.query(question)
        context = response.response
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    

custom_rag = RAG(query_engine)

question = "Give me detailed explaination about 1 bit LLms and how they work and the math behind it   "
pred = custom_rag(question)
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")



# lm.inspect_history(n= 3)

"""
Predicted Answer: **Detailed Explanation of 1-bit LLMS and How They Work**

As mentioned earlier, 1-bit LLMs (Low-precision Large Language Models) are a new concept that uses only one bit to represent each weight or activation value,
significantly reducing memory requirements and enabling faster and more efficient training and deployment on edge and mobile devices.

To understand how 1-bit LLMs work, we need to break down the process step by step. We'll explore the quantization techniques used, the reduction of activations from  
16 bits to 8 bits (1.58 bits), and the training methods employed.

**How 1-bit LLMS Work**

A 1-bit LLMS works by using a combination of quantization techniques to reduce the precision of the model's weights and activations. This is achieved through the     
following steps:

1. **Quantization**: The original 16-bit floating-point values are reduced to 8 bits (1.58 bits) using stochastic rounding or product quantization. Stochastic        
rounding involves randomly selecting one of the two closest representable values, while product quantization combines multiple weights into a single value.
2. **Activation reduction**: The activations from the previous layer are reduced in precision from 16 bits to 8 bits (1.58 bits) using stochastic rounding or product 
quantization.
3. **Training**: The model is trained on the reduced-precision data using a variant of stochastic gradient descent (SGD).

**Math Behind 1-bit LLMS**

The math behind 1-bit LLMs involves calculating the quantization error, stochastic rounding probability, and product quantization combined value.

* **Quantization Error**: The quantization error is calculated as the difference between the original floating-point value and its reduced-precision representation.  
This error can be minimized by using a suitable quantization technique.
* **Stochastic Rounding Probability**: The probability of selecting each value in stochastic rounding can be calculated using the following formula: P(x) = 0.5 \* (1 
+ sign(x - Q)), where x is the original value, Q is the quantization step size, and sign() is a function that returns -1 if x < 0 and 1 if x > 0.
* **Product Quantization Combined Value**: The combined value in product quantization can be calculated using the following formula: Q(x) = ∏(x_i / Q), where x_i are 
the individual weights, Q is the quantization step size, and ∏ denotes the product of the values.

**Conclusion**

1-bit LLMs work by using a combination of quantization techniques to reduce the precision of the model's weights and activations. This reduction in precision enables 
faster and more efficient training and deployment on edge and mobile devices. The math behind it involves calculating the quantization error, stochastic rounding     
probability, and product quantization combined value.

I hope this detailed explanation helps you understand how 1-bit LLMs work and the math behind it!
"""
