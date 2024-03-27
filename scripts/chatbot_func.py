# Import Packages:
import torch
from nltk.stem import PorterStemmer
from torch import nn
from text_cleaning import clean_text, bag_of_words
from model import NeuralNet
import random
import os

# write func to load model_params using torch.load


def laod_model_params(path: str):
    pass


def initialize_model(model_params: dict) -> NeuralNet:
    # Store the contents of the model dictionary:
    num_features = model_params["input_size"]
    hidden_layer_1 = model_params["hidden_size_1"]
    hidden_layer_2 = model_params["hidden_size_2"]
    num_classes = model_params["output_size"]

    # Load in an untrained model:
    model = NeuralNet(num_features, hidden_layer_1, hidden_layer_2, num_classes)

    # Change randomised model parameters to trained params:
    model.load_state_dict(model_params["model_weights"])
    # Set model to evaluation mode:
    model.eval()

    return model
