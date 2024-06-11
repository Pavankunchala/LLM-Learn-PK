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
from dsp.utils import deduplicate

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
Settings.llm =  Ollama(model="phi3")

ollama_local = dspy.OllamaLocal(model='phi3',max_tokens=10000,model_type="text",timeout_s= 125)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=ollama_local, rm=colbertv2_wiki17_abstracts)



document_directory = "m"
documents = SimpleDirectoryReader(document_directory).load_data()
# dspy.ColBERTv2.__doc__ =documents
# print(documents)
Settings.embed_model = resolve_embed_model("local:thenlper/gte-small")

#lets start indexing 
index = VectorStoreIndex.from_documents(documents, show_progress=True)

index.set_index_id("prop")
index.storage_context.persist("./propsely")

storgae_context = StorageContext.from_defaults(persist_dir="propsely")

index = load_index_from_storage(storage_context= storgae_context, index_id="prop")

# index = load_index_from_storage(storage_context= storgae_context, index_id="prop")
query_engine = index.as_query_engine()
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

class GenerateAnswer(dspy.Signature):
    """"Given Detailed answers and act as an SME and give detailed and more informative answers """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with relevant documents and citations , make it realy detailed and informative and organized  ")


 # PERForming Multi hop search for data 
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.query_engine = hyde_query_engine
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            response = self.query_engine.query(question)
            query = self.generate_query[hop](context=context, question=question).query
            passages = [response.response]
            # print(passages)
            context = deduplicate(context + passages)
        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
# Example usage
custom_rag = SimplifiedBaleen(hyde_query_engine)

question = "Give me detailed NOTES  of all the documents , make it so that you get detailed analysis of every part and divide them according to file name"
pred = custom_rag(question)
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")