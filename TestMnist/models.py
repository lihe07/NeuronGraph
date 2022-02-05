from torch import nn
import torch
from torch.autograd import Variable


def up_dim(x):
    """
    升维
    :param x:
    :return:
    """
    return x.reshape(1, -1)


class NG(nn.Module):
    def __init__(self, in_features, out_features, nodes=10):
        """
        初始化一个神经图
        :param nodes: 节点数量
        """
        super(NG, self).__init__()
        self.nodes_num = nodes
        self.nodes = nn.ModuleList([nn.Linear(nodes, 1) for _ in range(nodes)])
        assert nodes > (in_features + out_features), ValueError("全部节点数量不能少于输入输出节点数量")
        # 规定了输入节点的编号
        self.in_nodes = [i for i in range(in_features)]
        # 规定了输出节点的编号
        self.out_nodes = [nodes - i - 1 for i in range(out_features)]

    def tick(self, state: Variable) -> Variable:
        """
        一个tick，接受一个状态，执行推理，返回更新后的状态
        :param state: 模型当前的状态
        :return: 模型未来的状态
        """
        # 下一刻的状态
        next_state = Variable(torch.Tensor([]), requires_grad=True)
        for n in range(self.nodes_num):
            node = self.nodes[n]
            next_state = torch.cat((next_state, node(state)))

        return next_state

    def forward(self, x):
        """
        前向传播
        :param x:
        :return:
        """
        output = torch.Tensor([])
        # 修改以适应多batch
        for batch in x:
            state = torch.zeros(self.nodes_num)
            state[self.in_nodes] = batch
            state = Variable(state, requires_grad=True)
            next_state = self.tick(state)
            output = torch.cat((output, up_dim(next_state[self.out_nodes])))
        return output

    def parameters_num(self):
        """
        计算自身参数量
        :return:
        """
        return self.nodes_num ** 2


class LeNet(nn.Module):
    """
    LeNet PyTorch implementation
    for MNIST Dataset
    Input: [1x28x28]
    Layers:
        1. Conv 5x5 C: 1 -> 6
        2. Pooling 2x2
        3. Conv 5x5 C: 6 -> 16
        4. Pooling 2x2
        5. Conv 5x5 16 -> 120
        6. Softmax(Linear I:120 O: 10)
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5), padding=2),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, (5, 5)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 120, (5, 5)),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.layers(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class MnistNG(nn.Module):
    """
    NeuronGraph
    For MNIST Dataset
    输入：[Bx28x28]
    """

    def __init__(self):
        super(MnistNG, self).__init__()
        self.down_sample = nn.Conv2d(1, 1, (5, 5), stride=(3, 3)) # [Bx28x28] -> [Bx8x8]
        self.ng = NG(8*8, 10, nodes=100)

    def forward(self, x):
        x = x.reshape(x.size(0), 1, 28, 28)
        x = self.down_sample(x)
        x = x.view(x.size(0), -1)
        x = self.ng(x)
        return x
