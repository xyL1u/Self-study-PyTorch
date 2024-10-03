import torch
from torch.utils import data
from torchvision import datasets, transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# download datasets and transform
trans = transforms.ToTensor()
train_datasets = datasets.FashionMNIST(root='../FashionMNIST_data', train=True, transform=trans, download=True)
test_datasets = datasets.FashionMNIST(root='../FashionMNIST_data', train=False, transform=trans, download=True)

# create data loader
batch_size = 256
train_iter = data.DataLoader(train_datasets, batch_size, shuffle=True)
test_iter = data.DataLoader(test_datasets, batch_size, shuffle=False)

# model architecture
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                    nn.ReLU(), nn.Linear(256, 10))
loss = nn.CrossEntropyLoss()
lr = 0.03
trainer = torch.optim.SGD(params=net.parameters(), lr=lr)

# define confidence score
def confidence_scores(logits):

    probabilities = nn.functional.softmax(logits, dim=1)

    return probabilities

# define training loop for all epoch
def train_model(net, train_iter, test_iter, loss, num_epochs, trainer):

    train_losses = []
    train_accuracy = []
    test_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        net.train()

        # define training loop for one epoch
        for images, labels in train_iter:
            trainer.zero_grad()
            train_output = net(images)
            l = loss(train_output, labels)
            l.backward()
            trainer.step()  # update params
            running_loss += l.item()
            # compute accuracy
            _, predictor = torch.max(train_output.data, 1) # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predictor == labels).sum().item()

        epoch_loss = running_loss / len(train_iter)
        train_losses.append(epoch_loss)
        accuracy = 100 * correct / total
        train_accuracy.append(accuracy)
        test_accuracy = evaluate_model(net, test_iter)
        test_accuracies.append(test_accuracy)

    return train_losses, train_accuracy, test_accuracies

# define evaluate model
def evaluate_model(net, test_iter):

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_iter:
            train_output = net(images)
            _, predictor = torch.max(train_output.data, 1)
            total += labels.size(0)
            correct += (predictor == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

num_epochs = 10
train_losses, train_accuracy, test_accuracies = train_model(net, train_iter,test_iter, loss, num_epochs, trainer)

# Plot the training loss and train & test accuracy
plt.figure(figsize=(12, 5))

# Plot training loss
plt.plot(train_losses, label='Train Loss', linestyle='-')
# plt.plot(train_accuracy, label='Train Acc', marker='x', color='b')
# plt.plot(test_accuracies, label='Test Acc', marker='o',color='g')
plt.xlabel('Epoch')
plt.ylabel('Percentage / Loss')
plt.legend()
plt.show(block=False)

# Class names for FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize predictions on test set
def predictions(net, test_iter, n = 10):
    net.eval()
    images, labels = next(iter(test_iter)) # Get a batch of test images

    with torch.no_grad():
        train_output = net(images)
        _, predictor = torch.max(train_output.data, 1)
        confidence = confidence_scores(train_output)

    plt.figure(figsize=(12,6))

    for idx in range(n):
        plt.subplot(2, n // 2, idx+1)
        img = images[idx].numpy().squeeze()
        plt.imshow(img, cmap='gray')
        conf_scores = confidence[idx]
        conf_str = '\n'.join([f'{class_names[i]}:{conf_scores[i]:.2f}' for i in range(n)])
        plt.title(f'{class_names[predictor[idx].item()]} \n'
                  f'True:{class_names[labels[idx].item()]} \n'
                  f'Confidence scores:{conf_str}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

predictions(net, test_iter)