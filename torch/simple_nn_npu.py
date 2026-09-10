import torch
import torch.nn as nn
import torch.optim as optim
import torch_npu

# ============================================================
# NPU 适配说明：
# 1. 导入 torch_npu 后，torch.npu 命名空间即可用
# 2. 模型和数据通过 .npu() 方法迁移到 NPU
# 3. 其余训练逻辑（前向/反向/优化器）与 GPU 完全一致
# ============================================================

# 0. 选择 NPU 设备
device = torch.device("npu:0")
print(f"使用设备: {device}, NPU 数量: {torch_npu.npu.device_count()}")

# 1. 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)   # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)   # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 作为激活函数
        x = self.fc2(x)              # [修复] 原代码漏掉了参数 x
        return x

# 2. 创建模型实例并迁移到 NPU
model = SimpleNN().npu()
print(f"模型已在 NPU 上: {next(model.parameters()).device}")

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()            # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 假设有训练数据 X 和 Y（迁移到 NPU）
X = torch.randn(10, 2).npu()        # 10 个样本，2 个特征
Y = torch.randn(10, 1).npu()        # 10 个目标值

# 5. 训练循环
for epoch in range(100):
    optimizer.zero_grad()            # 清空之前的梯度
    output = model(X)                # 前向传播
    loss = criterion(output, Y)      # 计算损失
    loss.backward()                  # 反向传播
    optimizer.step()                 # 更新参数

    # 每 10 轮输出一次损失
    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

print("\n训练完成！")