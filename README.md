# Self-study-PyTorch
跟着沐神自学PyTorch，做一点简单的记录和存储。**欢迎大家指出表达错误或者代码错误**。
## 线性回归
- 线性回归从0开始（可查看不同learning rate对loss的影响）

- 线性回归简单表达

- Softmax从0开始

- Softmax简单表达（可查看test的结果）（Softmax只是将raw scores transform to probabilities,不能直接用于classification）
## 感知机（MPL）
- 感知机（把weight的初始化改为0了仍然work了，可能是因为数据集太简单了，用SGD就可以成功）

Perceptron_Classification是用单隐藏层感知机对FashionMNIST进行分类，可以输出loss function和confidence scores。  

总结：要先下载数据集并对图片进行处理，转换成tensor，并用dataloader对training dataset和test dataset“取样”，然后构造神经网络结构（包含loss function和optimization），构造一个train model对training data进行训练（backpropagation），再构造一个test model对test data进行检验。
- 权重衰退（regularization）
- kaggle的房价预测。（前两天因为要开学了就休息了两天，昨天开始调试的代码）

需要注意：1. 需要对特征进行筛选，不然内存会爆。
         2. pd.concat[]默认是axis=0,注意拼接方向。
         3. Cross-Validation代码要着重学习一下。

- 
