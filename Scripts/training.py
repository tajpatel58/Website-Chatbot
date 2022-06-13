# Import Packages:
import torch
import torch.nn as nn
import copy
import os
from chat_bot_dataset import ChatbotDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import NeuralNet
from torch.utils.tensorboard import SummaryWriter

#Create Writer:
writer = SummaryWriter()

#Initialise Training Dataset:
train_data = ChatbotDataset()

#Model Hyperparams: 
input_size = train_data.bag_size
hidden_layer_1 = 10
hidden_layer_2 = 6
num_classes = train_data.num_classses
batch_size = 6
learning_rate = 0.001
num_epochs = 1000

# Create Dataloader to pass data in batches:
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

#Initialise Model:
nn_model = NeuralNet(input_size, hidden_layer_1, hidden_layer_2, num_classes)

#Initialise Optimizer and loss(recall - CrossEntropy applies Softmax for us):
optimizer = Adam(nn_model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

#Training Loop:
def train(model, optim, loss_func, number_of_epochs=100):

    #Want to save the model with the highest accuracy on training set:
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(number_of_epochs):

        epoch_accuracy = 0
        model_state = copy.deepcopy(model.state_dict())

        for batch_no, (data, labels) in enumerate(train_loader):
            batch_no += 1
            predictions = model(data)
            error = loss_func(predictions, labels)
            _, class_predictions = torch.max(predictions, axis=1)
            epoch_accuracy += torch.sum(class_predictions == labels)
            writer.add_scalar('Loss/Train', error, epoch)
            error.backward()
            optim.step()
            optim.zero_grad()
        
        if epoch_accuracy >= best_acc: 
            best_model = model_state

    # Load the model with the highest testing accuracy
    model.load_state_dict(best_model)

    writer.flush()

    return model

trained_model = train(nn_model, optimizer, loss, number_of_epochs=1000)

# Once we complete the training loop, we want to store some of the parameters like the "bag" so that once the bot is built, 
# and a new message comes in, this can be turned into a feature vector and passed through our model. 
chat_model = {
    'net' : trained_model.state_dict(), 
    'input_size' : input_size, 
    'hidden_size_1' : hidden_layer_1, 
    'hidden_size_2' : hidden_layer_2, 
    'output_size' : num_classes, 
    'bag' : train_data.bag,
    'label_mapping' : train_data.label_mapping, 
    'raw_data' : train_data.raw_data
}

FILE = './Models/chat_model.pth'
torch.save(chat_model, FILE)
absolute_path = os.path.abspath('./Models/chat_model.pth')
print(f'Trained Model saved to {absolute_path}')