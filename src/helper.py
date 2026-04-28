import torch
import os
import numpy as np


class Helper:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.global_model = None
        self.clients = []

    def set_global_model(self, model):
        self.global_model = model.to(self.device)

    def set_clients(self, clients):
        self.clients = clients

    def save_client_update(self, client_id, state_dict, folder=None):
        folder = folder or f'{self.config.folder_path}/saved_updates'
        os.makedirs(folder, exist_ok=True)
        filepath = f'{folder}/update_{client_id}.pth'
        torch.save(state_dict, filepath)

    def save_foolsgold_history(self, client_id, state_dict, folder=None):
        folder = folder or f'{self.config.folder_path}/foolsgold'
        os.makedirs(folder, exist_ok=True)
        filepath = f'{folder}/history_{client_id}.pth'
        torch.save(state_dict, filepath)

    def load_client_update(self, client_id, folder=None):
        folder = folder or f'{self.config.folder_path}/saved_updates'
        filepath = f'{folder}/update_{client_id}.pth'
        if os.path.exists(filepath):
            return torch.load(filepath, map_location=self.device)
        return None

    def load_foolsgold_history(self, client_id, folder=None):
        folder = folder or f'{self.config.folder_path}/foolsgold'
        filepath = f'{folder}/history_{client_id}.pth'
        if os.path.exists(filepath):
            return torch.load(filepath, map_location=self.device)
        return None

    @property
    def num_sampled_participants(self):
        return self.config.num_sampled_participants

    @property
    def folder_path(self):
        return self.config.folder_path
