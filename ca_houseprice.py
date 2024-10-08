import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

train_data = pd.read_csv('california_houseprice/train.csv')
test_data = pd.read_csv('california_houseprice/test.csv')

# 整理训练数量和训练特征
n_train = train_data.shape[0]
train_f_process = pd.concat([train_data.iloc[:, 2], train_data.iloc[:, 5], train_data.iloc[:, 11:16]], axis=1)
test_f_process = pd.concat([test_data.iloc[:, 4], test_data.iloc[:, 10:15]], axis=1) # 对训练集和测试集一起做数据处理然后再分开，更简洁一点
all_features = pd.concat([train_f_process, test_f_process])

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features,dummy_na=True)
all_features = all_features * 1

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.iloc[:, 2].values.reshape(-1, 1), dtype=torch.float32)
# print(train_features.shape, train_labels.shape, test_features.shape)
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 256), nn.Dropout(0.5),
                        nn.ReLU(), nn.Linear(256, 64), nn.Dropout(0.5),
                        nn.ReLU(), nn.Linear(64, 1))
    return net

loss = nn.MSELoss()
device = torch.device('cuda')
net = get_net().to(device)

def data_loader(train_features, train_labels, batch_size, is_train=True):
    dataset = data.TensorDataset(train_features, train_labels)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=is_train)
    return dataloader

def log_rmse(net, features, labels):
    clamp_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clamp_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          batch_size, num_epoch, lr, wd):
    train_loss, test_loss = [], []
    train_iter = data_loader(train_features, train_labels, batch_size, is_train=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=wd)
    # print(f"Starting training with features shape: {train_features.shape} and labels shape: {train_labels.shape}")
    for epoch in range(num_epoch):
        net.train()
        for features, labels in train_iter:
            optimizer.zero_grad()
            l = loss(net(features), labels)
            l.backward()
            optimizer.step()
        train_loss.append(log_rmse(net, train_features, train_labels))
        # print(f'train loss{train_loss}, log rmse = {log_rmse(net, train_features, train_labels)}')
        if test_labels is not None:
            net.eval()
            test_loss.append(log_rmse(net, test_features, test_labels))
    return train_loss, test_loss

def get_k_fold_data(k, i, features, labels):
    assert k > 1
    fold_size = features.shape[0] // k
    features_train, labels_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        features_part, labels_part = features[idx, :], labels[idx]
        print(f"Fold {j}: X_part shape: {features_part.shape}, y_part shape: {labels_part.shape}")
        if j == i:
            features_valid, labels_valid = features_part, labels_part
            print(features_valid.shape, labels_valid.shape)
        elif features_train is None: # The first iteration isn't the validation fold
            features_train, labels_train = features_part, labels_part
            print(features_train.shape, labels_train.shape)
        else:
            features_train = torch.cat([features_train, features_part], 0)
            labels_train = torch.cat([labels_train, labels_part], 0)
            print(features_train.shape, labels_train.shape)
    # print(f"Fold {i}: Validation features shape: {features_valid.shape}, Validation labels shape: {labels_valid.shape}")
    return features_train, labels_train, features_valid, labels_valid

def k_fold(k, features_train, labels_train, batch_size, num_epoch, lr, wd):
    train_ls_sum, valid_ls_sum = 0, 0
    for i in range(k):
        features_train_fold, labels_train_fold, features_valid_fold, labels_valid_fold \
            = get_k_fold_data(k, i, features_train, labels_train)
        print(f'labels_valid:{labels_valid_fold.shape}')
        net = get_net()
        train_loss, valid_loss = train(net=net, train_features=features_train_fold, train_labels=labels_train_fold,
                                       test_features=features_valid_fold, test_labels=labels_valid_fold,
                                       batch_size=batch_size, num_epoch=num_epoch, lr=lr, wd=wd)
        train_ls_sum += train_loss[-1]
        valid_ls_sum += valid_loss[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epoch + 1)), [train_loss, valid_loss],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epoch],
                     legend=['train', 'valid'], yscale='log')
            plt.show()
        print(f'fold{i + 1}，train log rmse{float(train_loss[-1]):f}, '
              f'valid log rmse{float(valid_loss[-1]):f}')
    return train_ls_sum / k, valid_ls_sum / k

k, batch_size, num_epoch, lr, wd = 5, 256, 100, 0.5, 0
train_loss, valid_loss = k_fold(k, train_features, train_labels,
                                batch_size, num_epoch, lr, wd)

print(f'{k}-fold validation: training average log rmse: {float(train_loss):f},'
      f'validation average log rmse: {float(valid_loss):f}')