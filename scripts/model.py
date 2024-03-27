# Import Packages:
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size): 
        super().__init__()
        #Establishing Network architecture:
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size_1), 
                                    nn.ReLU(), 
                                    nn.Linear(hidden_size_1, hidden_size_2), 
                                    nn.ReLU(), 
                                    nn.Linear(hidden_size_2, output_size))
    
    def forward(self, x):
        # Feed datapoint through network, no-need to apply SoftMax.
        model_output = self.layers(x)
        return model_output