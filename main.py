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
from static_funcs import imshow, from_dataset_to_dataloader, get_model, compute_accuracy, training

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="lenet",
                    choices=["resnet18, lenet"])
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--aswa', action='store_true',
                    help='Adaptive SWA')
parser.add_argument('--swa', action='store_true',
                    help='SWA')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

args = parser.parse_args()

# Data
train_loader, val_loader, test_loader = from_dataset_to_dataloader(batch_size=args.batch_size, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net, best_acc, start_epoch = get_model(name=args.model, resume=args.resume)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

print("Training starts..")


def is_val_acc_increasing(last_val_running_model, val_ensemble_model, val_acc_running_model,tolerant=5,window=10) -> bool:
    assert tolerant>=0
    assert window >= tolerant

    val_acc_running_model.append(last_val_running_model)

    # (4) Does the validation performance of running model still increase?
    if val_ensemble_model is None:
        loss, loss_counter = 0.0, 0
        # (5.1) Iterate over the most recent 10 MRR scores
        for idx, acc in enumerate(val_acc_running_model[-window-1:]):
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


val_acc_running_model = []
val_acc_ensemble = None
sample_counter = None
apply_ensemble = True


def inplace_update_parameter_ensemble(ensemble_state_dict, current_model, sample_counter) -> None:
    with torch.no_grad():
        for k, parameters in current_model.state_dict().items():
            # (2) Update the parameter ensemble model with the current model.
            # Moving average
            ensemble_state_dict[k] = (ensemble_state_dict[k] * sample_counter + parameters) / (1 + self.sample_counter)


for epoch in range(start_epoch, args.num_epochs):

    start_time=time.time()
    train_loss = training(net, train_loader, loss_func=criterion, optimizer=optimizer)
    train_acc = compute_accuracy(net, train_loader)
    val_acc = compute_accuracy(net, val_loader)
    print(f"Epoch:{epoch} | RT: {time.time()-start_time:.2f} | Train Loss: {train_loss:.4f} | Train Acc.:{train_acc:.4f} | Val Acc.:{val_acc:.4f}")
    if apply_ensemble:
        if val_acc_ensemble is None and is_val_acc_increasing(last_val_running_model=val_acc,
                                                              val_ensemble_model=val_acc_ensemble,
                                                              val_acc_running_model=val_acc_running_model):
            """The validation performance of the running model is increasing"""
        else:
            if val_acc_ensemble is None:
                print("Initialize parameter ensemble")
                torch.save(net.state_dict(), f=f"{self.path}/trainer_checkpoint_main.pt")
                val_acc_ensemble = val_acc
                sample_counter += 1
                continue

            print("Update parameter ensemble")
            ensemble_state_dict = torch.load(f"{args.model}_checkpoint.pt", torch.device(net.device))
            # Update
            inplace_update_parameter_ensemble(ensemble_state_dict, net, sample_counter=sample_counter)
            # Evaluate
            ensemble = type(net)
            ensemble.load_state_dict(ensemble_state_dict)
            mrr_updated_ensemble_model = compute_accuracy(ensemble, val_loader)
            if mrr_updated_ensemble_model>val_acc_ensemble:
                print("STORE")
            else:
                "Do not store"

x, y = next(iter(test_loader))
x = x[:3]
y = y[:3]
# print images
# imshow(torchvision.utils.make_grid(x))
print('Y: ', ' '.join(f'{classes[y[j]]:5s}' for j in range(len(y))))
net.eval()
with torch.no_grad():
    predicted_classes = torch.argmax(net(x), 1).tolist()
print('Yhat: ', ' '.join(f'{classes[j]:5s}' for j in predicted_classes))
