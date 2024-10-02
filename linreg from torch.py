import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

# Generate dataset para
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# Generate synthetic dataset
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # Feature
    y = torch.matmul(X, w) + b # Linear transformation
    y += torch.normal(0, 0.01, y.shape) #Add noise
    return X, y.reshape((-1, 1)) # -1 means figure out this dimension based on the size of the table

# Create 1000 examples
features, labels = synthetic_data(true_w, true_b, 1000)

# Create data iteration
def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    # DataLoader for random selecting from dataset
batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

from torch import nn
net = nn.Sequential(nn.Linear(2, 1)) # Define linear model

#Initialization params
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.001)

num_epochs = 20
loss_values = []
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step() # Update model
    l = loss(net(features), labels)
    loss_values.append(l.mean().item())
    print(f'epoch{epoch + 1}, loss {l:f}')

plt.plot(range(1, num_epochs + 1), loss_values, marker = 'o', color = 'b')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()