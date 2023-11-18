'''Train CIFAR10 with PyTorch.'''
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import os
import argparse
import time
# from utils import *
from static_funcs import imshow, from_dataset_to_dataloader, get_model, compute_accuracy, training, save_running_net, \
    init_aswa, is_val_acc_increasing, init_swa

from ensemblers import ASWA, SWA

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="lenet",
                    choices=["resnet18, lenet"])
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--aswa', action='store_true',
                    help='Adaptive SWA')
parser.add_argument('--swa', action='store_true', help='SWA')
parser.add_argument('--early_stooping', action='store_true', help='early_stooping')
parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--batch_size', default=1000, type=int)
parser.add_argument('--num_epochs', default=1, type=int)

args = parser.parse_args()
print("Config:", args)
# Data
train_loader, val_loader, test_loader = from_dataset_to_dataloader(batch_size=args.batch_size, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.resume is None:
    flag = None
    if args.swa:
        flag = "_SWA"
    elif args.aswa:
        flag = "_ASWA"
    elif args.early_stooping:
        flag = "_ES"
    else:
        flag = ""
    checkpoint_name = f"./checkpoint_{args.model}{flag}"
    resume = False
else:
    checkpoint_name = args.resume
    resume = True

net, start_epoch = get_model(name=args.model, checkpoint=checkpoint_name, resume=resume)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
print("Training starts..")


swa = None
aswa = None
train_acc_running = []
val_acc_running_model = []
val_acc_running = []
# checkwheter we have a checkpoint for ensemble
if args.aswa and args.resume and os.path.exists(f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt"):
    aswa = ASWA(path=f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt", model_name=args.model)
    val_acc_ensemble = aswa.val_acc_ensemble
    sample_counter = aswa.sample_counter
elif args.swa and args.resume and os.path.exists(f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt"):
    swa = SWA(path=f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt", model_name=args.model)
    sample_counter = swa.sample_counter
else:
    val_acc_ensemble = None
    sample_counter = 0


for epoch in range(start_epoch, args.num_epochs):

    start_time = time.time()
    train_loss = training(net, train_loader, loss_func=criterion, optimizer=optimizer)
    train_acc = compute_accuracy(net, train_loader)
    val_acc = compute_accuracy(net, val_loader)

    train_acc_running.append(train_acc)
    val_acc_running.append(val_acc)

    save_running_net(checkpoint_name, name=args.model, net=net, train_loss=train_loss, train_acc=train_acc,
                     val_acc=val_acc, epoch=epoch)
    print(f"Epoch:{epoch} | RT: {time.time() - start_time:.2f} | Train Loss: {train_loss:.4f} "
          f"| Train Acc.:{train_acc:.4f} | Val Acc.:{val_acc:.4f}", end="")
    if args.early_stooping:
        if is_val_acc_increasing(last_val_running_model=val_acc,
                                 val_ensemble_model=val_acc_ensemble,
                                 val_acc_running_model=val_acc_running_model) is False:
            break
    elif args.aswa:
        if val_acc_ensemble is None and is_val_acc_increasing(last_val_running_model=val_acc,
                                                              val_ensemble_model=val_acc_ensemble,
                                                              val_acc_running_model=val_acc_running_model):
            """The validation performance of the running model is increasing"""
        else:
            if val_acc_ensemble is None:
                print(" | ASWA ensemble initialized...")
                val_acc_ensemble = val_acc
                sample_counter += 1
                init_aswa(checkpoint=checkpoint_name, name=args.model, state_dict=net.state_dict(),
                          val_acc=val_acc_ensemble,
                          sample_counter=sample_counter, epoch=epoch)
                continue
            else:
                # ASWA load
                aswa = ASWA(path=f"{checkpoint_name}/ASWA_{args.model}_checkpoint.pt", model_name=args.model)
                aswa.forward(epoch, net, val_acc_net=val_acc,
                             val_loader=val_loader)  # look_ahead and update_if_necessary
    elif args.swa:
        if swa is None:
            sample_counter += 1
            init_swa(checkpoint, name=args.model, state_dict=net.state_dict(), sample_counter=sample_counter,
                     epoch=epoch)

        swa = SWA(f"{checkpoint_name}/SWA_{args.model}_checkpoint.pt", model_name=args.model)
        swa.forward(epoch, net, val_loader=val_loader)
    else:
        """Pass"""

    print("")

## TESTING

if args.aswa:
    aswa = ASWA(path=f"./checkpoint/ASWA_{args.model}_checkpoint.pt", model_name=args.model)
    test_acc = compute_accuracy(aswa.get_net(net), test_loader)
    print(f"Test ASWA {args.model}:", test_acc)
elif args.swa:
    aswa = SWA(path=f"./checkpoint/SWA_{args.model}_checkpoint.pt", model_name=args.model)
    test_acc = compute_accuracy(swa.get_net(net), test_loader)
    print(f"Test SWA {args.model}:", test_acc)
else:
    # if args.early_stooping:
    net, _ = get_model(name=args.model, checkpoint=checkpoint_name, resume=True)

print(
    f"Train Acc.:{compute_accuracy(net, train_loader):.4f} | Val Acc.:{compute_accuracy(net, val_loader):.4f}| Test Acc.:{compute_accuracy(net, test_loader):.4f}")

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
