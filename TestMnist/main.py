from rich.progress import track
from rich.console import Console
import torch
from models import MnistNG, LeNet
from torch.autograd import Variable

c = Console()


def get_classes(out):
    """
    通过模型的概率输出获取类别
    :param out:
    :return:
    """
    classes = []
    for i in range(out.size(0)):
        classes.append(torch.argmax(out[i]))
    return classes


def test(model, test_loader):
    model.eval()
    correct_num = 0
    for inputs, labels in track(test_loader, description=f"Test..."):
        with torch.no_grad():
            output = model(inputs)
            result = get_classes(output)
            if result[0] == labels[0]:
                correct_num += 1

    c.log("Accuracy: {}".format(correct_num / len(test_loader)))


# 定义训练函数
def train(model):
    """
    基线训练
    :return:
    """
    # 加载数据
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    from torch.optim.lr_scheduler import StepLR
    # 定义数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义优化器
    optimizer = Adam(model.parameters(), lr=0.001)
    # 定义学习率衰减
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    # 定义损失函数
    loss_func = CrossEntropyLoss()
    # 定义训练周期
    epochs = 10
    # 定义训练循环
    for epoch in range(epochs):
        model.train(True)
        loss_sum = 0
        for inputs, labels in track(train_loader, description=f"Epoch-{epoch}..."):
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, labels)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        c.log("Epoch-{} Loss: {}".format(epoch, loss_sum / len(train_loader)))

        if epoch % 2 == 0:
            test(model, test_loader)

        scheduler.step()


if __name__ == '__main__':
    my_model = MnistNG()
    my_model.ng.init()
    print(my_model.ng.get_adjacency_matrix())
    exit()
    try:
        train(my_model)
    except KeyboardInterrupt:
        c.log("Interrupted!")
