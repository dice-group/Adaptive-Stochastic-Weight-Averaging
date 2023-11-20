'''Train CIFAR10 with PyTorch.'''
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import argparse
import time
from static_funcs import imshow, from_dataset_to_dataloader, init_running_model, compute_accuracy, training, \
    save_running_net, \
    init_aswa, is_val_acc_increasing, init_swa, gen_checkpoint_name, get_optim

from ensemblers import ASWA, SWA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="lenet",
                    choices=["resnet18", "lenet"])
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--aswa', action='store_true', help='Adaptive SWA')
parser.add_argument('--swa', action='store_true', help='SWA')
parser.add_argument('--early_stooping', action='store_true', help='early_stooping')
parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--batch_size', default=1000, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--optim', default="Adam", choices=["Adam", "SGD"], type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--plot', action="store_true")


args = parser.parse_args()
print("Config:", args)
# Data
train_loader, val_loader, test_loader = from_dataset_to_dataloader(batch_size=args.batch_size,
                                                                   num_workers=args.num_workers)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
checkpoint_name, resume = gen_checkpoint_name(args)

# (1) Initialize or load a running model
net = init_running_model(name=args.model)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = get_optim(optim_name=args.optim, params=net.parameters(), lr=args.lr)
print("Training starts..")
swa = None
aswa = None

if not os.path.isdir(checkpoint_name):
    os.mkdir(checkpoint_name)

# Load parameters of running model
if args.resume:
    assert os.path.isdir(checkpoint_name)
    checkpoint = torch.load(f'{checkpoint_name}/{args.model}_checkpoint.pt')
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_acc_running = checkpoint["train_acc"]
    val_acc_running = checkpoint["val_acc"]
    assert len(checkpoint["val_acc"]) == len(checkpoint["train_acc"])
    start_epoch = len(train_acc_running)
    del checkpoint
else:
    if not os.path.isdir(checkpoint_name):
        os.mkdir(checkpoint_name)
    start_epoch = 0
    train_acc_running = []
    val_acc_running = []

# Check whether  we have a checkpoint for ensemble
if args.aswa and args.resume and os.path.exists(f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt"):
    print("Loading ASWA...")
    aswa_state = torch.load(f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt")
    train_acc_aswa = aswa_state["train_acc"]
    val_acc_aswa = aswa_state["val_acc"]
    size_aswa = aswa_ensemble_state["size"]
elif args.swa and args.resume and os.path.exists(f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt"):
    print("Loading SWA...")
    swa_state = torch.load(f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt")
    train_acc_swa = swa_state["train_acc"]
    val_acc_swa = swa_state["val_acc"]
    size_swa = swa_ensemble_state["size"]
else:
    train_acc_aswa = []
    val_acc_aswa =  []

    train_acc_swa =  []
    val_acc_swa =  []


for epoch in range(start_epoch, args.num_epochs):
    start_time = time.time()
    train_loss = training(net, train_loader, loss_func=criterion, optimizer=optimizer)

    train_acc_running.append(compute_accuracy(net, train_loader))
    val_acc_running.append(compute_accuracy(net, val_loader))

    torch.save({"net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_acc": train_acc_running, "val_acc": val_acc_running,
                "epoch": epoch},
               f=f"{checkpoint_name}/{args.model}_checkpoint.pt")

    print(f"Epoch:{epoch} | RT: {time.time() - start_time:.2f} "
          f"| Train Loss: {train_loss:.4f} "
          f"| Train Acc.:{train_acc_running[-1]:.4f} "
          f"| Val Acc.:{val_acc_running[-1]:.4f}", end=" ")

    if args.aswa:
        if len(val_acc_aswa) == 0 or val_acc_running[-1] > val_acc_aswa[-1]:
            train_acc_aswa.append(train_acc_running[-1])
            val_acc_aswa.append(val_acc_running[-1])
            torch.save({"net": net.state_dict(), "train_acc": train_acc_aswa, "val_acc": val_acc_aswa,
                        "size": 1, "epoch": epoch}, f=f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt")
        else:
            # Reject or Soft Update
            ensemble_state = torch.load(f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt")
            ensemble_state_dict = ensemble_state["net"]
            with torch.no_grad():
                for k, parameters in net.state_dict().items():
                    ensemble_state_dict[k] = (ensemble_state_dict[k] * ensemble_state["size"] + parameters) / (
                            1 + ensemble_state["size"])
            # Look-head Eval
            ensemble_net = type(net)()
            ensemble_net.load_state_dict(ensemble_state_dict)
            val_provisional = compute_accuracy(ensemble_net, val_loader)
            if val_provisional > ensemble_state["val_acc"][-1]:
                train_acc_aswa.append(compute_accuracy(net, train_loader))
                val_acc_aswa.append(val_provisional)
                torch.save({"net": net.state_dict(),
                            "train_acc": train_acc_aswa,
                            "val_acc": val_acc_aswa,
                            "size": ensemble_state["size"] + 1,
                            "epoch": epoch}, f=f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt")
        print(f"| ASWA Val Acc: {val_acc_aswa[-1]}", end=" ")

    if args.swa:
        if len(val_acc_swa) == 0:
            train_acc_swa.append(train_acc_running[-1])
            val_acc_swa.append(val_acc_running[-1])
            torch.save({"net": net.state_dict(), "train_acc": train_acc_swa, "val_acc": val_acc_swa,
                        "size": 1, "epoch": epoch}, f=f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt")
        else:
            # Reject or Soft Update
            ensemble_state = torch.load(f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt")
            ensemble_state_dict = ensemble_state["net"]
            with torch.no_grad():
                for k, parameters in net.state_dict().items():
                    ensemble_state_dict[k] = (ensemble_state_dict[k] * ensemble_state["size"] + parameters) / (
                            1 + ensemble_state["size"])
            # Look-head Eval
            ensemble_net = type(net)()
            ensemble_net.load_state_dict(ensemble_state_dict)
            train_acc_swa.append(compute_accuracy(net, train_loader))
            val_acc_swa.append(compute_accuracy(ensemble_net, val_loader))

            torch.save({"net": net.state_dict(),
                        "train_acc": train_acc_swa,
                        "val_acc": val_acc_swa,
                        "size": ensemble_state["size"] + 1,
                        "epoch": epoch}, f=f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt")
        print(f"| SWA Val Acc: {val_acc_swa[-1]}",end="")
    else:
        """Pass"""

    print("")
## TESTING

print(f"Test base {args.model}:", compute_accuracy(net, test_loader))
if args.plot:
    plt.plot(train_acc_running, label="Train Acc-Base")
    plt.plot(val_acc_running, label="Val Acc-Base")

if args.aswa:
    ensemble_state = torch.load(f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt")
    ensemble_state_dict = ensemble_state["net"]
    ensemble_net = type(net)()
    ensemble_net.load_state_dict(ensemble_state_dict)
    test_acc = compute_accuracy(ensemble_net, test_loader)
    print(f"Test ASWA {args.model}:", test_acc)
    if args.plot:
        plt.plot(ensemble_state["train_acc"], label="Train Acc-ASWA")
        plt.plot(ensemble_state["val_acc"], label="Val Acc-ASWA")

if args.swa:
    ensemble_state = torch.load(f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt")
    ensemble_state_dict = ensemble_state["net"]
    ensemble_net = type(net)()
    ensemble_net.load_state_dict(ensemble_state_dict)
    test_acc = compute_accuracy(ensemble_net, test_loader)
    print(f"Test SWA {args.model}:", test_acc)
    if args.plot:
        plt.plot(ensemble_state["train_acc"], label="Train Acc-SWA")
        plt.plot(ensemble_state["val_acc"], label="Val Acc-SWA")

if args.plot:
    plt.legend()
    plt.show()

print("First 3 Test data points")
x, y = next(iter(test_loader))
x = x[:3]
y = y[:3]
# print images
# imshow(torchvision.utils.make_grid(x))
print('T: ', ' '.join(f'{classes[y[j]]:5s}' for j in range(len(y))))
net.eval()
with torch.no_grad():
    predicted_classes = torch.argmax(net(x), 1).tolist()
print('P: ', ' '.join(f'{classes[j]:5s}' for j in predicted_classes))
