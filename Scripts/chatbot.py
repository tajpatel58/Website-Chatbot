#Import Packages:
import torch
from nltk.stem import PorterStemmer
from torch import nn
from text_cleaning import clean_text, bag_of_words
from model import NeuralNet
from ts.torch_handler.base_handler import BaseHandler
import random
import os


class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self._context = None
        self.model = None


    def initialize(self, context):
        #Context passed in during deployment
        self._context = context
        self.initialized = True
        self.manifest = context.manifest

        properties = context.system_properties
        #Construct path for .pth file
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pth_path = os.path.join(model_dir, serialized_file)

        # Load in helper variables:
        self.model_params = torch.load(model_pth_path)
        
        #Store the contents of the model dictionary:
        self.num_features = self.model_params['input_size']
        hidden_layer_1 = self.model_params['hidden_size_1']
        hidden_layer_2 = self.model_params['hidden_size_2']
        self.num_classes = self.model_params['output_size']
        self.bag = self.model_params['bag']
        self.label_mapping = self.model_params['label_mapping']
        self.raw_data = self.model_params['raw_data']

        #Load in an untrained model:
        self.model = NeuralNet(self.num_features, hidden_layer_1, hidden_layer_2, self.num_classes)

        #Change randomised model parameters to trained params:
        self.model.load_state_dict(self.model_params['model_weights'])
        # Set model to evaluation mode:
        self.model.eval()

        #Initialise Stemmer:
        self.stem = PorterStemmer()


        ### Function to take in a message as text and output a response: 
    def preprocess(self, data):
        print(data)
        message = data[0].get("body").get("message")
        clean_message = clean_text(message, self.stem)
        feature_vec = bag_of_words(clean_message, self.bag)
        # Reshape into a matrix
        feature_vec = feature_vec.reshape(1, self.num_features)
        return feature_vec


    def inference(self, ftrs_vec):
        #Feed through model:
        output_vec = self.model(ftrs_vec)
        return output_vec


    def postprocess(self, output_vec):
        index_to_tag = {v : k for k,v in self.label_mapping.items()}
        with torch.no_grad():
            probabilities_vec = torch.softmax(output_vec, 1)
        tag_probabilities = {index_to_tag.get(i) : probabilities_vec[0, i].item() for i in range(self.num_classes)}
        return [tag_probabilities]


    def respond(self, output_vec):
        val, class_num = torch.max(output_vec, axis=1)
        predicted_tag = list(self.label_mapping.keys())[class_num]
        probability_class = torch.softmax(val, axis=0)
        if probability_class >= 0.75:
            for message_group in self.raw_data['messages']:
                if predicted_tag == message_group['tag']:
                    random_response = random.choice(message_group['responses'])
                    return random_response
        else:
            return "I'm not sure what you mean, please try a different question"


    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        response = self.respond(model_output)
        tag_probabilities = self.postprocess(model_output)
        return [response]