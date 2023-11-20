import matplotlib.pyplot as plt
import torch


running_checkpoint = torch.load(f'checkpoint_lenet_SWA/lenet_checkpoint.pt')

aswa_checkpoint = torch.load(f'checkpoint_lenet_SWA/ASWA_lenet_checkpoint.pt')

swa_running_checkpoint = torch.load(f'checkpoint_lenet_SWA/SWA_lenet_checkpoint.pt')

plt.plot(running_checkpoint["train_acc"], label="Train Base")
plt.plot(running_checkpoint["val_acc"], label="Val Base")

plt.plot(aswa_checkpoint["train_acc"], label="Train ASWA")
plt.plot(aswa_checkpoint["val_acc"], label="Val ASWA")

plt.plot(swa_running_checkpoint["train_acc"], label="Train SWA")
plt.plot(swa_running_checkpoint["val_acc"], label="Val SWA")


plt.ylabel("Acc.")
plt.xlabel("Epochs")
plt.legend()
plt.show()