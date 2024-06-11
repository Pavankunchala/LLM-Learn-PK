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



lm = dspy.OllamaLocal(model='llama3',max_tokens=10000,model_type="text",timeout_s= 125)
Settings.llm =  Ollama(model="llama3")
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)


document_directory = "m"
documents = SimpleDirectoryReader(document_directory).load_data()
# print(documents)
Settings.embed_model = resolve_embed_model("local:thenlper/gte-small")

#lets start indexing 
index = VectorStoreIndex.from_documents(documents, show_progress=True)

index.set_index_id("prop")
index.storage_context.persist("./propsely")

storgae_context = StorageContext.from_defaults(persist_dir="propsely")

index = load_index_from_storage(storage_context= storgae_context, index_id="prop")
query_engine = index.as_query_engine(response_mode ="tree_summarize")

#creating the signature 
class DocSummarizer(Signature):
    """From the given document and create a detailed summary of the document, using the retrived context for accuracy"""
    context = InputField(desc="May contain relevant facts from retrieved documents")
    question = InputField(desc="The question that needs an answer")
    answer = OutputField(desc="Give detailed answers and more information about the answer and relevant links if necessary")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.query_engine = query_engine
        self.generate_answer = ChainOfThought(DocSummarizer)
        print("Class 2 created")

    def forward(self, question):
        response = self.query_engine.query(question)
        context = response.response
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)



custom_rag = RAG(query_engine)

question = "Give me detailed summary of all the documents and dividide accordingly to their file name"
pred = custom_rag(question)
print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")