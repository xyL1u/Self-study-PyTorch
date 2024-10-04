import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# def l1_penalty(w):
#     return torch.sum(abs(w))

def l2_penalty(w):
    return torch.sum(w ** 2 / 2)

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003

    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l= loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f'Epoch {epoch + 1}, Train loss: {train_loss}, Test loss: {test_loss}')
    plt.figure(figsize=(0, 6))
    plt.plot(range(5, num_epochs + 1, 5), train_losses, label='Train Loss')
    plt.plot(range(5, num_epochs + 1, 5), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing')
    plt.legend()
    plt.show()
    print('w的L2范数是：', torch.norm(w).item())
train(lambd=200)