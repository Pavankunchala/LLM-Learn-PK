
from datasets import load_dataset
from datasets import load_dataset_builder

dataset = load_dataset_builder('flytech/python-codes-25k')


#in this code we are trying to just load the dataset 
print(dataset.info.description+"\n")

#to see its features
print(dataset.info.features)

#if you like the featues then you can load the dataset like this 
# it is also important to split its features 
dataset = load_dataset('flytech/python-codes-25k', split='train')


# One can map the dataset in any way, for the sake of example:
dataset = dataset.map(lambda example: {'text': example['instruction'] + ' ' + example['input'] + ' ' + example['output']})['text']

# print(dataset[0:25])