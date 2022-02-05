import time
from copy import deepcopy

from torch import nn
import torch
from icecream import ic
from torch import optim
from torch.autograd import Variable


def up_dim(x: Variable) -> Variable:
    return x.reshape(1, *x.shape)


class Node(nn.Module):
    def __init__(self, neighbors):
        super(Node, self).__init__()
        self.weight = nn.Parameter(torch.rand(neighbors), requires_grad=True)

    def forward(self, x: Variable) -> Variable:
        out = torch.zeros(x.shape[1:])
        out = Variable(out, requires_grad=True)
        for i, data in enumerate(x):
            out = out + data * self.weight[i]
        return out


class NG(nn.Module):
    def __init__(self, nodes=10):
        """
        初始化一个神经图
        :param nodes: 节点数量
        """
        super(NG, self).__init__()
        self.nodes_num = nodes
        self.nodes = nn.ModuleList([Node(nodes) for _ in range(nodes)])
        self.in_node = 0
        self.out_node = nodes - 1

    def tick(self, state: Variable) -> Variable:
        """
        一个tick，接受一个状态，执行推理，返回更新后的状态
        :param state: 模型当前的状态
        :return: 模型未来的状态
        """
        next_state = Variable(torch.Tensor([]), requires_grad=True)
        for n in range(self.nodes_num):
            node = self.nodes[n]
            out = node(state)
            next_state = torch.cat((next_state, up_dim(out)))

        return next_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x:
        :return:
        """
        state = torch.zeros((self.nodes_num,) + tuple(x.shape))
        ic(state.shape)
        state[self.in_node] = x
        state = Variable(state, requires_grad=True)
        next_state = self.tick(state)
        return next_state[self.out_node]

    def parameters_num(self):
        """
        计算自身参数量
        :return:
        """
        return self.nodes_num ** 2


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 18),
            nn.Linear(18, 18),
            nn.Linear(18, 9)
        )

    def forward(self, x):
        return self.layers(x.reshape(9)).reshape(3, 3)


def test():
    """
    loss不变: 状态无法backward
    :return:
    """
    my_graph = NG(100)
    ic(my_graph.parameters_num())
    loss_func = nn.MSELoss()
    input_ = torch.Tensor([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    target = torch.Tensor([
        [1, 1, 4],
        [5, 1, 4],
        [19, 19, 810]
    ])
    target_state = torch.ones(100, 3, 3)
    prs = [_ for _ in my_graph.parameters()]
    ic(len(prs))

    optimizer = optim.SGD(my_graph.parameters(), 0.1, momentum=0.8)
    for i in range(10):
        optimizer.zero_grad()
        out = my_graph(input_)
        loss = loss_func(out, target)
        ic(loss)
        loss.backward()
        optimizer.step()
    ic(out)


def test_baseline():
    model = Baseline()
    input_ = torch.Tensor([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    target = torch.Tensor([
        [1, 1, 4],
        [5, 1, 4],
        [19, 19, 810]
    ])
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), 0.001)
    for i in range(10):
        optimizer.zero_grad()
        out = model(input_)
        loss = loss_func(out, target)
        ic(loss)
        loss.backward()
        optimizer.step()


def test_node():
    """
    Test Passed
    """
    model = Node(3)
    input_ = torch.Tensor([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    target = torch.Tensor([1, 1, 4])
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), 0.1)
    for i in range(10):
        optimizer.zero_grad()
        out = model(input_)
        loss = loss_func(out, target)
        ic(loss)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    test()
