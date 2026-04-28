import torch
import os


class Config:
    def __init__(self, **kwargs):
        # Dataset settings
        self.dataset = kwargs.get('dataset', 'mnist')
        self.data_dir = kwargs.get('data_dir', './data')
        self.num_classes = kwargs.get('num_classes', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        self.test_batch_size = kwargs.get('test_batch_size', 100)

        # Model settings
        self.model = kwargs.get('model', 'cnn')

        # Federated learning settings
        self.num_clients = kwargs.get('num_clients', 100)
        self.num_sampled_participants = kwargs.get('num_sampled_participants', 10)
        self.num_malicious_clients = kwargs.get('num_malicious_clients', 0)
        self.attack_type = kwargs.get('attack_type', 'none')
        self.aggr_epochs = kwargs.get('aggr_epochs', 1)

        # Aggregation settings
        self.agg_method = kwargs.get('agg_method', 'avg')
        self.clip_factor = kwargs.get('clip_factor', 1.0)

        # Training settings
        self.lr = kwargs.get('lr', 0.01)
        self.momentum = kwargs.get('momentum', 0.9)
        self.local_epochs = kwargs.get('local_epochs', 5)
        self.global_epochs = kwargs.get('global_epochs', 100)

        # System settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.folder_path = kwargs.get('folder_path', './results')
        self.seed = kwargs.get('seed', 42)

        # APRA specific
        self.apra_pca_components = kwargs.get('apra_pca_components', 3)
        self.apra_base_clip = kwargs.get('apra_base_clip', 1.0)
        self.apra_k_init = kwargs.get('apra_k_init', 5.0)
        self.apra_k_decay = kwargs.get('apra_k_decay', 0.1)
        self.apra_use_neup_ddif = kwargs.get('apra_use_neup_ddif', True)

        self._create_dirs()

    def _create_dirs(self):
        os.makedirs(self.folder_path, exist_ok=True)
        os.makedirs(f'{self.folder_path}/foolsgold', exist_ok=True)
        os.makedirs(f'{self.folder_path}/saved_updates', exist_ok=True)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)
