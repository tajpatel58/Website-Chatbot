# Import relevant packages:
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
from Scripts.text_cleaning import clean_text, bag_of_words

class ChatbotDataset(Dataset):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.load_data()
        self.clean_data()


    def __getitem__(self, idx): 
        cleaned_text = self.cleaned_messages_list[idx]
        features_vector = bag_of_words(cleaned_text, self.bag)
        corresponding_encoded_label = self.numeric_labels[idx]
        return features_vector, corresponding_encoded_label


    def __len__(self):
        return len(self.messages_list)
    

    def get_raw_data(self, data_path):
        with open(data_path, 'r') as f:
            message_json = json.loads(f.read())

        return message_json


    def load_data(self): 
        self.raw_data = self.get_raw_data('/Users/tajsmac/Documents/Girlfriend-Chatbot/Data/message_data.json')

        # Dictionary to store numerical encoding of labels.
        self.label_mapping = {}
        #label_index increments everytime a new label/tag is seen. 
        label_index = 0

        # list to store each message in dataset:
        self.messages_list = []

        #list to store message_labels:
        self.labels = []
        
        for message_group in self.raw_data['messages']:
            # create a numerical encoding for message tags.
            if message_group['tag'] not in self.label_mapping:
                self.label_mapping[message_group['tag']] = label_index
                label_index += 1
            
            self.messages_list.extend(message_group['keywords'])
            self.labels.extend([message_group['tag']] * len(message_group['keywords']))
        
        # Get numerical encoding of labels
        self.numeric_labels = [self.label_mapping[label] for label in self.labels]

        #Number of classes:
        self.num_classses = label_index


    def clean_data(self):
        # Clean up all messages and extracting "bag" for bag of words:
        self.bag = set()

        # list to hold cleaned messages:
        self.cleaned_messages_list = []
        for message in self.messages_list:
            cleaned_message = clean_text(message, self.stemmer)
            self.cleaned_messages_list.append(cleaned_message)
            self.bag.update(set(cleaned_message))
        self.bag = list(self.bag)
        self.bag_size = len(self.bag)