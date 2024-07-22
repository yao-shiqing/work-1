import torch
import torch.nn as nn

class SL_Network(nn.Module):
    def __init__(self, n, j, k):
       super(SL_Network, self).__init__()
       self.n = n
       self.j = j
       self.input_dim = n * k
       self.fc1 = nn.Linear(self.input_dim, n)
       self.fc2 = nn.Linear(n, j)
       self.fc3 = nn.Linear(j, n * j)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        output = x3.view(-1, self.n, self.j)
        return output



# test
# network = SL_Network(n=10, j=5, k=3)
# input_sample = torch.randn(100,10,3)
# output = network(input_sample)
# print("输入大小:", input_sample.size())
# print("输出大小:", output.size())
