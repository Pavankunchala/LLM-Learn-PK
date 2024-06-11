from llama_index.core import VectorStoreIndex,get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from langchain_community.llms import Ollama
from rich import print
from dspy import (
    Signature,
    InputField,
    OutputField,
    Predict,
    ChainOfThought
)
import dspy
Settings.llm =  Ollama(model="phi3")

ollama_local = dspy.OllamaLocal(model='phi3',max_tokens=10000,model_type="text",timeout_s= 125)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=ollama_local, rm=colbertv2_wiki17_abstracts)

# give the document directory here 

document_directory = "data"
documents = SimpleDirectoryReader(document_directory).load_data()
# dspy.ColBERTv2.__doc__ =documents
# print(documents)
Settings.embed_model = resolve_embed_model("local:thenlper/gte-small")

#lets start indexing 
index = VectorStoreIndex.from_documents(documents, show_progress=True)

index.set_index_id("test")
index.storage_context.persist("./testing")

storgae_context = StorageContext.from_defaults(persist_dir="testing")

index = load_index_from_storage(storage_context= storgae_context, index_id="test")

retriver = VectorIndexRetriever(index = index, similarity_top_k=6)
response_syth = get_response_synthesizer()

query_engine = RetrieverQueryEngine(retriever= retriver,response_synthesizer= response_syth)

# 
class GenerateAnswer(Signature):
    """Give Detailed and infomrative answers, Make sure they are correct, it is really important for me that information is correct """

    context = InputField(desc ="Containts the information you need to read from ")
    question = InputField()
    answer = OutputField(desc= "Give Detailed Information , in clear and correct format, with citations and make it extremely detailed and informative")



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

question = "Give me detailed NOTES  of all the documents , make it so that you get detailed analysis of every part and divide them according to file name"
pred = custom_rag(question)
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")


    
