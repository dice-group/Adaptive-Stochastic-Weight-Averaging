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
parser.add_argument('--aswa', action='store_false',
                    help='Adaptive SWA')
parser.add_argument('--swa', action='store_true',
                    help='SWA')
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', default=8102, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

args = parser.parse_args()

# Data
train_loader, val_loader, test_loader = from_dataset_to_dataloader(batch_size=args.batch_size, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net, start_epoch = get_model(name=args.model, resume=args.resume)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

print("Training starts..")


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


val_acc_running_model = []
val_acc_ensemble = None
sample_counter = 0


class ASWA:
    def __init__(self, path: str):
        assert os.path.exists(path)
        self.ensemble_state = torch.load(path)
        self.val_acc_ensemble = self.ensemble_state["val_acc"]
        self.sample_counter = self.ensemble_state["sample_counter"]

    def inplace_update_parameter_ensemble(self, current_model):
        with torch.no_grad():
            for k, parameters in current_model.state_dict().items():
                # (2) Update the parameter ensemble model with the current model.
                # Moving average
                self.ensemble_state_dict[k] = (self.ensemble_state_dict[k] * self.sample_counter + parameters) / (
                            1 + self.sample_counter)

    def get_net(self,net):
        # Evaluate
        ensemble_net = type(net)()
        ensemble_net.load_state_dict(self.ensemble_state_dict)
        return ensemble_net

    def save(self,epoch, val_acc_ensemble):
        self.sample_counter += 1
        self.val_acc_ensemble=val_acc_ensemble
        ensemble_state = {
            'aswa_ensemble': self.ensemble_state_dict,
            "val_acc": self.val_acc_ensemble,
            "sample_counter": self.sample_counter,
            'epoch': epoch,
        }
        print(
            f"UPDATE Parameter Ensemble | Sample_counter: {ensemble_state['sample_counter']} | Val Acc.:{ensemble_state['val_acc']:.4f}")
        print(f"")
        torch.save(ensemble_state, f=f"./checkpoint/ASWA_{args.model}_checkpoint.pt")

# checkwheter we have a checkpoint for ensemble
if args.aswa and args.resume:
    aswa = ASWA(path=f"./checkpoint/ASWA_{args.model}_checkpoint.pt")
    val_acc_ensemble = aswa.val_acc_ensemble
    sample_counter = aswa.sample_counter

for epoch in range(start_epoch, args.num_epochs):

    start_time = time.time()
    train_loss = training(net, train_loader, loss_func=criterion, optimizer=optimizer)
    train_acc = compute_accuracy(net, train_loader)
    val_acc = compute_accuracy(net, val_loader)
    # Save the checkpoint
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save({
        'net': net.state_dict(),
        'train_loss': train_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        'epoch': epoch,
    }, f=f"./checkpoint/{args.model}_checkpoint.pt")

    print(f"Epoch:{epoch} | RT: {time.time() - start_time:.2f} | Train Loss: {train_loss:.4f} "
          f"| Train Acc.:{train_acc:.4f} | Val Acc.:{val_acc:.4f}")
    if args.aswa:
        if val_acc_ensemble is None and is_val_acc_increasing(last_val_running_model=val_acc,
                                                              val_ensemble_model=val_acc_ensemble,
                                                              val_acc_running_model=val_acc_running_model):
            """The validation performance of the running model is increasing"""
        else:
            if val_acc_ensemble is None:
                print("Initialize parameter ensemble...")
                val_acc_ensemble = val_acc
                sample_counter += 1
                state = {'aswa_ensemble': net.state_dict(), "val_acc": val_acc_ensemble,
                         "sample_counter": sample_counter, 'epoch': epoch}
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, f=f"./checkpoint/ASWA_{args.model}_checkpoint.pt")
                continue

            aswa = ASWA(path=f"./checkpoint/ASWA_{args.model}_checkpoint.pt")
            # Look ahead
            aswa.inplace_update_parameter_ensemble(net)

            updated_val_acc_ensemble_model = compute_accuracy(aswa.get_net(net), val_loader)

            if updated_val_acc_ensemble_model > val_acc_ensemble:
                aswa.save(epoch,updated_val_acc_ensemble_model)
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
