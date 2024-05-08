
from datasets import load_dataset
from datasets import load_dataset_builder

dataset1 = load_dataset_builder('flytech/python-codes-25k')


#in this code we are trying to just load the dataset 
print(dataset1.info.description+"\n")
# print(type(dataset1))

#to see its features
print(dataset1.info.features)

#if you like the featues then you can load the dataset like this 
# it is also important to split its features 
dataset = load_dataset('flytech/python-codes-25k', split='train')
# print(dataset[0])

#indexing the data by colums
# print(dataset[0]["text"])

# the Order of the indexing matters for large datasets so its bettrer to go with row first and column second
"""
text = dataset[0]["text"] this is faster than 
text = dataset["text"][0] tnis 
"""


# One can map the dataset in any way, for the sake of example:
# dataset = dataset.map(lambda example: {'text': example['instruction'] + ' ' + example['input'] + ' ' + example['output']})['text']

# print(type(dataset))

data_dict = {idx: data for idx, data in enumerate(dataset)}

# Now, let's verify what we have done by printing the first item in the dictionary
# print(data_dict[0])

# To enhance understanding, let's inspect the type of this dictionary
# print(type(data_dict))

# If you want to access a specific column across all entries (like 'text'), you can do it using:
texts = {idx: data['text'] for idx, data in enumerate(dataset)}
inputs ={idx: data['input'] for idx, data in enumerate(dataset)}
instructions = {idx: data['instruction'] for idx, data in enumerate(dataset)}
outputs = {idx: data['output'] for idx, data in enumerate(dataset)}


print(inputs[0]+ "\n")

print(instructions[1]+ "\n")

print(outputs[0]+ "\n")


# Now print the first text to see what it looks like
print(texts[1])

