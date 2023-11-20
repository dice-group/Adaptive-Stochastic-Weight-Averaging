import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
from models import *
import torch.optim as optim
from torch.utils.data import random_split


def gen_checkpoint_name(args):
    if args.resume is None:
        if args.swa:
            flag = "_SWA"
        elif args.aswa:
            flag = "_ASWA"
        elif args.early_stooping:
            flag = "_ES"
        else:
            flag = ""
        checkpoint_name = f"checkpoint_{args.model}{flag}"
        resume = False
    else:
        checkpoint_name = args.resume
        resume = True

    return checkpoint_name, resume


def get_optim(optim_name, params, lr):
    if optim_name == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    else:
        optimizer = optim.SGD(params, lr=lr)

    return optimizer


def is_val_acc_increasing(last_val_running_model, val_ensemble_model, val_acc_running_model, tolerant=3,
                          window=6) -> bool:
    assert tolerant >= 0
    assert window >= tolerant

    val_acc_running_model.append(last_val_running_model)

    # (4) Does the validation performance of running model still increase?
    if val_ensemble_model is None:
        loss, loss_counter = 0.0, 0
        # (5.1) Iterate over the most recent 10 MRR scores
        for idx, acc in enumerate(val_acc_running_model[-window - 1:]):
            if last_val_running_model > acc:
                """No loss"""
            else:
                loss_counter += 1

        if loss_counter >= tolerant:
            return False
        else:
            return True
    else:
        return False


def init_aswa(checkpoint, name, state_dict, val_acc, sample_counter, epoch):
    state = {'aswa_ensemble': state_dict, "val_acc": val_acc,
             "sample_counter": sample_counter, 'epoch': epoch}
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, f=f"{checkpoint}/ASWA_{name}_checkpoint.pt")


def init_swa(checkpoint, name, state_dict, sample_counter, epoch):
    state = {'swa_ensemble': state_dict, "sample_counter": sample_counter, 'epoch': epoch}
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, f=f"{checkpoint}/SWA_{name}_checkpoint.pt")


def save_running_net(checkpoint, name, net, train_loss, train_acc, val_acc, epoch):
    # Save the checkpoint
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)

    torch.save({'net': net.state_dict(), 'train_loss': train_loss, "train_acc": train_acc, "val_acc": val_acc,
                'epoch': epoch}, f=f"{checkpoint}/{name}_checkpoint.pt")


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


def init_running_model(name):
    model = None
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
    return model
