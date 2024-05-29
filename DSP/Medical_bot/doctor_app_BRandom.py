import dspy
from dsp.utils import deduplicate
import streamlit as st
import numexpr as ne

ne.set_num_threads(28)


lm = dspy.OllamaLocal(model='llama3',max_tokens=4000,model_type="text",timeout_s= 250)
# metricLM = dspy.OllamaLocal(model='phi3', max_tokens=4000, model_type='text')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)



# Load data

# dl = DataLoader()
# train_dataset = []
# code_alpaca = dl.from_huggingface("lavita/ChatDoctor-HealthCareMagic-100k", split=["train"])
# train_dataset, dataset_test = [], []
# for split in ['train']:
#     for example in code_alpaca[split]:
#         dsp_example = dspy.Example(question=example['input'], answer=example['output']).with_inputs("question")

#         if split == 'train':
#             train_dataset.append(dsp_example)
    


#now lets create a basic rag function 

class GenerateAnswer(dspy.Signature):
    """Act as a Amazing Doctor with broad medical knowledge, and give detailed Diagnostic the with links and remedy if required  based on the  symptoms and problems given by the User """

    context = dspy.InputField(desc="Helpful information for answering the question.")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Give Detailed Answers to problem presented by the user with Links and Medication if required based on the diagnostic act as DOCTOR")

class RAG(dspy.Module):
    def __init__(self, num_passages=4):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    
model = RAG()  # 




model.load('doctor_Brandom.json')


# lm.inspect_history(n=1)

st.title("Medical chatbot with BootStrapRandomSearch in Dspy ")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.container():
    for message in st.session_state["messages"]:
        st.info(f"{message['role'].title()}: {message['content']}")

user_input = st.text_input("Ask your question:", key="user_input")
if st.button("Submit"):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    user_input = user_input+"act as a Doctor and give diagnostics and remedies with Medicences" 
    prediction = model.forward(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": prediction.answer})
    st.experimental_rerun()



#make a note i am pretty sure there is more optimized way to stream the output in str