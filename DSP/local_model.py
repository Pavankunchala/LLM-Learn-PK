import dspy


#in here we are using a Local Lama model 
# You can download Ollama from here https://ollama.com/

#this assumes that you have already installed ollama and downloaded llama3


lm = dspy.OllamaLocal(model='llama3')
dspy.settings.configure(lm=lm)


# you can directly give an input if you want to, but usually its not recommended

response = lm("Hello how are you today")

print(response)