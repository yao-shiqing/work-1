import torch
import torch.nn as nn

class SL_Network(nn.Module):
    def __init__(self, n, j, k):
        super(SL_Network, self).__init__()

        # 输入维度为n个obs_k矩阵的组合，假设obs_k矩阵展平后为一维数组
        self.input_dim = n * k  # 输入维度为n * k
        # 第一层全连接层，输入维度为n * k，输出维度为n
        self.fc1 = nn.Linear(self.input_dim, n)
        # 第二层全连接层，输入维度为n，输出维度为j
        self.fc2 = nn.Linear(n, j)

    def forward(self, x):
        """
        前向传播函数
        参数：
        x: 输入张量，形状为 (batch_size, n, k)

        返回：
        输出张量，形状为 (batch_size, n * j)
        """
        # 将输入展平为 (batch_size, n * k)
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # 将输出形状调整为 (batch_size, n * j)
        output = x.view(x.size(0), -1)

        return output


# 定义网络参数
n = 5  # 输入维度的n
j = 3  # 第二层全连接层的输出维度j
# 假设obs_k的形状为 (batch_size, n, k)
batch_size = 2
k = 55
input_data = torch.randn(batch_size, n, k)

# 获取obs_k的大小k
k = input_data.size(2)

model = SL_Network(n, j, k)
print(model)
output = model(input_data)
print(output.shape)  # 预期形状为 (batch_size, n * j)
