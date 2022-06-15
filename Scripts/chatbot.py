#Import Packages:
import torch
import json
from Scripts.model import NeuralNet
from Scripts.text_cleaning import clean_text, bag_of_words
from nltk.stem import PorterStemmer
from torchserve.torch_handler.base_handler import BaseHandler
import random

class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

        
    def initialize(self):
        # Load in helper variables:
        self.helper_var = torch.load('/Users/tajsmac/Documents/Website-Chatbot/Models/chat_model_helpers.pth')
        
        #Store the contents of the model dictionary:
        self.num_features = self.helper_var['input_size']
        hidden_layer_1 = self.helper_var['hidden_size_1']
        hidden_layer_2 = self.helper_var['hidden_size_2']
        num_classes = self.helper_var['output_size']
        self.bag = self.helper_var['bag']
        self.label_mapping = self.helper_var['label_mapping']
        self.raw_data = self.helper_var['raw_data']

        #Load in an untrained model:
        self.model = NeuralNet(self.num_features, hidden_layer_1, hidden_layer_2, num_classes)

        #Change randomised model parameters to trained params:
        self.model.load_state_dict(self.chatbot_model_weights)

        # Set model to evaluation mode:
        self.model.eval()

        #Initialise Stemmer:
        self.stem = PorterStemmer()

        ### Function to take in a message as text and output a response: 
    def preprocess(self, message):
        clean_message = clean_text(message, self.stem)
        feature_vec = bag_of_words(clean_message, self.bag)
        # Reshape into a matrix
        feature_vec = feature_vec.reshape(1, self.num_features)
        return feature_vec


    def inference(self, ftrs_vec):
        #Feed through model:
        output_vec = self.model(ftrs_vec)
        # Based on the fact Softmax function is an increasing function, the index of highest value is the class we're predicting,
        val, prediction = torch.max(output_vec, axis=1)
        return val, prediction

    def handle(self, val, prediction):
    
        # Note that the variable "prediction" is a label number, to get the actual label/tag we can use the label,mapping dictionary. 
        predicted_tag = list(self.label_mapping.keys())[prediction]
        # Only give a response if we're more than 75% sure that the tag is correct (ie the probabilitiy of this datapoint belonging to class is >= 0.75)
        probability = torch.softmax(val, axis=0)
        if probability >= 0.75:
            # Choose a random response from the predefined responses:
            for message_group in self.raw_data['messages']:
                if predicted_tag == message_group['tag']:
                    random_response = random.choice(message_group['responses'])
                    return random_response
        else:
            return "I'm not sure what you mean, please try a different message. :)"
