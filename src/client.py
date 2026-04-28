import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


class Client:
    def __init__(self, client_id, train_loader, helper, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.helper = helper
        self.device = device
        self.model = None
        self.is_malicious = False
        self.attack_type = None

    def set_model(self, model):
        self.model = copy.deepcopy(model).to(self.device)

    def set_malicious(self, attack_type='label_flip'):
        self.is_malicious = True
        self.attack_type = attack_type

    def local_train(self, epochs=5, lr=0.01, momentum=0.9, criterion=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                if self.is_malicious and self.attack_type == 'label_flip':
                    target = self._label_flip(target)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

            avg_loss = total_loss / total_samples
            accuracy = 100. * total_correct / total_samples

        return self.model.state_dict(), avg_loss, accuracy

    def _label_flip(self, targets):
        num_classes = self.helper.config.num_classes
        return (num_classes - 1 - targets) % num_classes

    def compute_gradient_update(self, global_state_dict, local_state_dict, lr=0.01):
        update = {}
        for name in global_state_dict:
            if name in local_state_dict:
                update[name] = (local_state_dict[name] - global_state_dict[name]) / lr
        return update


class WeightAccumulator:
    def __init__(self, global_model):
        self.accumulator = {}
        for name, data in global_model.state_dict().items():
            self.accumulator[name] = torch.zeros_like(data)

    def add_update(self, update):
        for name in self.accumulator:
            if name in update:
                self.accumulator[name] += update[name]

    def get(self, name=None):
        if name:
            return self.accumulator.get(name)
        return self.accumulator

    def zero(self):
        for name in self.accumulator:
            self.accumulator[name].zero_()
