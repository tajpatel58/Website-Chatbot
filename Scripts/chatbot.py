#Import Packages:
import torch
import json
from Scripts.model import NeuralNet
from Scripts.text_cleaning import clean_text, bag_of_words
from nltk.stem import PorterStemmer
import random

class Chatbot:
    def __init__(self, path):
    # Load in the trained model:
        self.chatbot_model_info = torch.load(path)

        #Store the contents of the model dictionary:
        self.num_features = self.chatbot_model_info['input_size']
        hidden_layer_1 = self.chatbot_model_info['hidden_size_1']
        hidden_layer_2 = self.chatbot_model_info['hidden_size_2']
        num_classes = self.chatbot_model_info['output_size']
        self.bag = self.chatbot_model_info['bag']
        self.label_mapping = self.chatbot_model_info['label_mapping']
        trained_params = self.chatbot_model_info['net']
        self.raw_data = self.chatbot_model_info['raw_data']

        #Load in an untrained model:
        self.net = NeuralNet(self.num_features, hidden_layer_1, hidden_layer_2, num_classes)

        #Change randomised model parameters to trained params:
        self.net.load_state_dict(trained_params)

        # Set model to evaluation mode:
        self.net.eval()

        #Initialise Stemmer:
        self.stem = PorterStemmer()

        ### Function to take in a message as text and output a response: 
    def respond(self, message):
        clean_message = clean_text(message, self.stem)
        feature_vec = bag_of_words(clean_message, self.bag)
        # Reshape into a matrix
        feature_vec = feature_vec.reshape(1, self.num_features)
        #Feed through model:
        output_vec = self.net(feature_vec)
        # Based on the fact Softmax function is an increasing function, the index of highest value is the class we're predicting,
        val, prediction = torch.max(output_vec, axis=1)
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

    def runbot(self):
        while True:
            message = input('Radhika: ')
            if message == 'quit':
                print('Goodbye xx')
                break
            print(f'Taj: {self.respond(message)}')


bot = Chatbot('/Users/tajsmac/Documents/Girlfriend-Chatbot/Models/chat_model.pth').runbot()
