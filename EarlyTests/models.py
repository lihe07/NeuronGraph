import time
from copy import deepcopy

from torch import nn
import torch
from icecream import ic
from torch import optim
from torch.autograd import Variable


def up_dim(x: Variable) -> Variable:
    return x.reshape(1, *x.shape)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x:
        :return:
        """
        state = torch.zeros(self.nodes_num)
        ic(state.shape)
        state[self.in_nodes] = x
        state = Variable(state, requires_grad=True)
        next_state = self.tick(state)
        return next_state[self.out_nodes]

    def parameters_num(self):
        """
        计算自身参数量
        :return:
        """
        return self.nodes_num ** 2


def test():
    my_graph = NG(2, 5, nodes=100)
    loss_func = nn.MSELoss()

