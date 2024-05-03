import copy
import nltk
import torch
import mlflow
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from torch.utils.data import DataLoader
from src.scripts.models.neural_net import NeuralNet
from src.scripts.data.dataset import ChatbotDataset
from torch.utils.tensorboard import SummaryWriter
nltk.download('stopwords')

# Create Writer:
writer = SummaryWriter("/Users/taj/Documents/Website-Chatbot/logging/tensorboard")
mlflow.set_tracking_uri("http://127.0.0.1:6090")
mlflow.set_experiment("chatbot")

# Data Processing Variables:
data_path = Path("/Users/taj/Documents/Website-Chatbot/data/message_data.json")
preprocessing_objs_path = Path("/Users/taj/Documents/Website-Chatbot/logging/processing/")
stemmer = PorterStemmer()
stop_words = list(stopwords.words("english"))

# Initialise Training Dataset:
train_data = ChatbotDataset(data_path, stemmer, stop_words)
train_data.load_and_process_data()
preprocessing_artifacts_paths = train_data.pickle_preprocessing_objs(preprocessing_objs_path)

# Model Hyperparams:
input_size = train_data.get_bag_size()
hidden_layer_1 = 10
hidden_layer_2 = 6
num_classes = train_data.get_num_classes()
batch_size = 6
learning_rate = 0.001
num_epochs = 1000
num_training_data_points = len(train_data)

# Create Dataloader to pass data in batches:
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# Initialise Model:
nn_model = NeuralNet(input_size, hidden_layer_1, hidden_layer_2, num_classes)

# Initialise Optimizer and loss(recall - CrossEntropy applies Softmax for us):
optimizer = Adam(nn_model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


# Training Loop:
def train(model, optim, loss_func, number_of_epochs=100):
    # Want to save the model with the highest accuracy on training set:
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(number_of_epochs):
        print(f"training epoch: {epoch}")
        epoch_accuracy = 0
        model_state = copy.deepcopy(model.state_dict())

        for batch_no, (data, labels) in enumerate(train_loader):
            batch_no += 1
            predictions = model(data)
            error = loss_func(predictions, labels)
            _, class_predictions = torch.max(predictions, axis=1)
            epoch_accuracy += torch.sum(class_predictions == labels)
            writer.add_scalar("Loss/Train", error, epoch)
            error.backward()
            optim.step()
            optim.zero_grad()

        epoch_accuracy = torch.div(epoch_accuracy, num_training_data_points)
        if epoch_accuracy >= best_acc:
            best_model = model_state
            best_acc = epoch_accuracy

    # Load the model with the highest testing accuracy
    model.load_state_dict(best_model)

    writer.flush()

    return model, best_acc


trained_model, best_accuracy = train(nn_model, optimizer, loss, number_of_epochs=1000)
mlflow.pytorch.log_model(trained_model, "model", 
                         code_paths=preprocessing_artifacts_paths)
mlflow.log_metric("training_loss", best_accuracy)

# Once we complete the training loop, we want to store some of the parameters like the "bag" so that once the bot is built,
# and a new message comes in, this can be turned into a feature vector and passed through our model.
