import torch
import torch.nn as nn
import torch.optim as optim

# 1.定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2) # 输出层到隐藏层
        self.fc2 = nn.Linear(2, 1) # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # RELU作为激活函数
        x = self.fc2(x)
        return x

# 2.创建模型实例
model = SimpleNN()

# 3.定义损失函数和优化器
critertion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam优化器

# 4.假设有训练数据x和y
X = torch.randn(10, 2) # 10个样本，2个特征
Y = torch.randn(10, 1) # 10个目标值

# 5. 训练循环
for epoch in range(100):  # 训练100轮
    optimizer.zero_grad() # 清空之前的梯度
    output = model(X)  # 前向传播
    loss = critertion(output, Y) #计算损失
    loss.backward() # 反向传播
    optimizer.step() # 更新参数

    # 每10轮输出一次损失
    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')


