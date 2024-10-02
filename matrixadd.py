import torch
import random
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

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad(): # Temporarily disable gradient computation.PyTorch will automatically track the updates
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() # Reset the gradients


# Traverse the dataset and sample. Shuffle dataset and batch them
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, "\n", y)
    break

# Define hyperparams
num_epochs = 20
net = linreg
loss = squared_loss

def train_model(lr, num_epochs, batch_size, features, labels):
    # Initialize the params
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    loss_values = []
    print(f'\nTraining with learning rate = {lr}\n')
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            # print(f'Params at update at epoch {epoch + 1}; w.grad = {w.grad}， b.grad = {b.grad}')
            sgd([w, b], lr, batch_size)
            # print(f'Params after update at epoch {epoch + 1}; w = {w}， b = {b}')
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            loss_values.append(train_l.mean().item())
            print(f"epoch{epoch + 1}, loss{float(train_l.mean()):f}")
    return loss_values

lr1 = 0.01
loss_values_1 = train_model(lr1, num_epochs, batch_size, features, labels)

lr2 = 0.001
loss_values_2 = train_model(lr2, num_epochs, batch_size, features, labels)

lr3 = 0.0001
loss_values_3 = train_model(lr3, num_epochs, batch_size, features, labels)

plt.plot(range(1, num_epochs + 1), loss_values_1, marker = 'o', color = 'b')
plt.plot(range(1, num_epochs + 1), loss_values_2, marker = 'x', color = 'g')
plt.plot(range(1, num_epochs + 1), loss_values_3, marker = 'o', color = 'r')
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()
