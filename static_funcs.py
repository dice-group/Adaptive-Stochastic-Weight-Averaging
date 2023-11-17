import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
from models import *

from torch.utils.data import random_split


def training(net, loader, loss_func, optimizer) -> float:
    net.train()
    sum_of_minibatch__loss = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        sum_of_minibatch__loss += loss.item()
    return sum_of_minibatch__loss


def compute_accuracy(model, loader):
    # Compute Accuracy
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for x, y in loader:
            # the class with the highest energy is what we choose as prediction
            _, yhat = torch.max(model(x).data, 1)
            total += y.size(0)
            correct += (yhat == y).sum().item()
    return correct / total


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def from_dataset_to_dataloader(batch_size=1024, num_workers=32, seed=1):
    # Define ttran
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    torch.manual_seed(seed)
    assert len(train_dataset) == 50000
    train_size, val_size = 40_000, 10_000

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_model(name, resume: bool):
    model = None
    acc = 0
    epoch = 0
    if name == "lenet":
        model = LeNet()
    elif name == "resnet18":
        model = ResNet18()
    elif name == "resnet152":
        model = ResNet152()
    elif name == "vgg":
        model = VGG("VGG19")
    elif name == "densenet":
        model = DenseNet201()

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint')
        checkpoint = torch.load(f'./checkpoint/{name}_checkpoint.pt')
        model.load_state_dict(checkpoint['net'])
        epoch = checkpoint['epoch']

    return model, epoch
