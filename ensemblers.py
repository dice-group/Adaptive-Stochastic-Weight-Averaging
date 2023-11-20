from static_funcs import compute_accuracy
import torch


class ASWA:
    def __init__(self, path: str, model_name: str):
        self.ensemble_state = torch.load(path)
        self.running_model_name = model_name
        self.ensemble_state_dict = self.ensemble_state["aswa_ensemble"]
        self.val_acc_ensemble = self.ensemble_state["val_acc"]
        self.sample_counter = self.ensemble_state["sample_counter"]

    def forward(self, epoch, net, val_acc_net, val_loader):
        updated_val_acc_ensemble_model = self.look_ahead(net, val_loader)
        print(
            f" | Val Param Ensemble: {self.val_acc_ensemble:.4f} | Look-ahead Param Ensemble: {updated_val_acc_ensemble_model:.4f}",
            end=" | ")
        if val_acc_net > updated_val_acc_ensemble_model and val_acc_net > self.val_acc_ensemble:
            # hard update
            ensemble_state = {
                'aswa_ensemble': net.state_dict(),
                "val_acc": val_acc_net,
                "sample_counter": 1,
                'epoch': epoch,
            }
            print(f"Hard update", end="\t")
            torch.save(ensemble_state, f=f"checkpoint/ASWA_{self.running_model_name}_checkpoint.pt")
        elif updated_val_acc_ensemble_model > self.val_acc_ensemble:
            self.sample_counter += 1
            self.val_acc_ensemble = updated_val_acc_ensemble_model
            ensemble_state = {
                'aswa_ensemble': self.ensemble_state_dict,
                "val_acc": self.val_acc_ensemble,
                "sample_counter": self.sample_counter,
                'epoch': epoch,
            }
            print(f"Soft Update", end="\t")
            torch.save(ensemble_state, f=f"checkpoint/ASWA_{self.running_model_name}_checkpoint.pt")
        else:
            print(f"Reject Update", end="\t")

    def get_net(self, net):
        ensemble_net = type(net)()
        ensemble_net.load_state_dict(self.ensemble_state_dict)
        return ensemble_net

    def inplace_update_parameter_ensemble(self, current_model):
        with torch.no_grad():
            for k, parameters in current_model.state_dict().items():
                # (2) Update the parameter ensemble model with the current model.
                # Moving average
                self.ensemble_state_dict[k] = (self.ensemble_state_dict[k] * self.sample_counter + parameters) / (
                        1 + self.sample_counter)

    def look_ahead(self, net, val_loader) -> float:
        # Parameter Look ahead
        self.inplace_update_parameter_ensemble(net)
        ensemble_net = type(net)()
        ensemble_net.load_state_dict(self.ensemble_state_dict)
        return compute_accuracy(ensemble_net, val_loader)


class SWA:
    def __init__(self, path: str, model_name: str):
        self.ensemble_state = torch.load(path)
        self.running_model_name = model_name
        self.ensemble_state_dict = self.ensemble_state["swa_ensemble"]
        self.sample_counter = self.ensemble_state["sample_counter"]

    def forward(self, epoch, net, val_loader):
        self.inplace_update_parameter_ensemble(net)
        ensemble_net = type(net)()
        ensemble_net.load_state_dict(self.ensemble_state_dict)

        updated_val_acc_ensemble_model = compute_accuracy(ensemble_net, val_loader)
        print(f" | Val SWA Ensemble: {updated_val_acc_ensemble_model:.4f}", end="")
        self.sample_counter += 1
        ensemble_state = {
            'swa_ensemble': self.ensemble_state_dict,
            "sample_counter": self.sample_counter,
            'epoch': epoch}
        torch.save(ensemble_state, f=f"checkpoint/SWA_{self.running_model_name}_checkpoint.pt")

    def get_net(self, net):
        ensemble_net = type(net)()
        ensemble_net.load_state_dict(self.ensemble_state_dict)
        return ensemble_net

    def inplace_update_parameter_ensemble(self, current_model):
        with torch.no_grad():
            for k, parameters in current_model.state_dict().items():
                # (2) Update the parameter ensemble model with the current model.
                # Moving average
                self.ensemble_state_dict[k] = (self.ensemble_state_dict[k] * self.sample_counter + parameters) / (
                        1 + self.sample_counter)
