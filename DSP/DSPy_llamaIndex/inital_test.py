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



lm = dspy.OllamaLocal(model='gemma:2b',max_tokens=4000,model_type="text",timeout_s= 125)
Settings.llm =  Ollama(model="gemma:2b")
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
query_engine = index.as_query_engine(response_mode="tree_summarize")

# Create signature
class GenerateAnswer(Signature):
    """Answer questions with detailed answers, using retrieved context for accuracy."""
    context = InputField(desc="May contain relevant facts from retrieved documents")
    question = InputField(desc="The question that needs an answer")
    answer = OutputField(desc="Give detailed answers and more information about the answer")
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

question = "What are 1 bit llms give me detailed explaination like i am kid   "
pred = custom_rag(question)
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")



# lm.inspect_history(n= 3)
