import dspy
from dsp.utils import deduplicate
import streamlit as st

lm = dspy.OllamaLocal(model='llama3:8b-instruct-q5_1', max_tokens=4000)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

#you can train it with ur own dataset you can refere to multi hop custom.py in the same folder 



# Define the signatures


class GenerateAnswer(dspy.Signature):
    """Answer questions with detailed answers and code."""
    context = dspy.InputField(desc="May contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed output with code if required")

class RAG(dspy.Module):
    def __init__(self, num_passages=4):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="May contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)

# Initialize the model
model = SimplifiedBaleen()
model.load('multihop.json')

# Streamlit app
st.title("Multi hop coding chatbot with DSPY ")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.container():
    for message in st.session_state["messages"]:
        st.info(f"{message['role'].title()}: {message['content']}")

user_input = st.text_input("Ask your question:", key="user_input")
if st.button("Submit"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    prediction = model.forward(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": prediction.answer})
    st.experimental_rerun()



#make a note i am pretty sure there is more optimized way to stream the output in streamlit, so if you have any suggestions do let me know 