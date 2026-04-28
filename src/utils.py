import torch
import numpy as np
import random
import os
import sys
from collections import OrderedDict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config):
    import logging
    log_dir = config.folder_path
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy


def compute_model_diff(model_a, model_b):
    diff = OrderedDict()
    for name, data in model_a.state_dict().items():
        if name in model_b.state_dict():
            diff[name] = data - model_b.state_dict()[name]
    return diff


def flatten_update(update_dict, layer_names=None):
    flat = []
    for name, data in update_dict.items():
        if layer_names and not any(l in name for l in layer_names):
            continue
        if 'num_batches_tracked' in name:
            continue
        flat.append(data.cpu().numpy().flatten())
    if flat:
        return np.concatenate(flat)
    return np.array([])


def get_layer_name_for_dataset(dataset):
    if dataset in ['cifar10', 'cifar100']:
        return 'linear'
    elif dataset == 'tiny-imagenet-200':
        return 'fc'
    else:
        return 'fc2'


def compute_cosine_similarity_matrix(updates):
    n = len(updates)
    cs = np.zeros((n, n))
    for i in range(n):
        norm_i = np.linalg.norm(updates[i]) + 1e-8
        for j in range(n):
            norm_j = np.linalg.norm(updates[j]) + 1e-8
            cs[i, j] = np.dot(updates[i], updates[j]) / (norm_i * norm_j)
    return cs


def gap_statistic(data, num_sampling=5, K_max=10, n=None):
    from sklearn.cluster import KMeans
    if n is None:
        n = len(data)
    W_k = np.zeros(K_max)
    W_kb = np.zeros((num_sampling, K_max))
    sk = np.zeros(K_max)

    for k in range(1, K_max + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
        kmeans.fit(data)
        W_k[k - 1] = np.sum([np.sum((data[kmeans.labels_ == i] -
                                      kmeans.cluster_centers_[i]) ** 2)
                             for i in range(k)])

        for b in range(num_sampling):
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            random_data = np.random.uniform(min_vals, max_vals, size=data.shape)
            kmeans_b = KMeans(n_clusters=k, init='k-means++', n_init=10)
            kmeans_b.fit(random_data)
            W_kb[b, k - 1] = np.sum([np.sum((random_data[kmeans_b.labels_ == i] -
                                              kmeans_b.cluster_centers_[i]) ** 2)
                                     for i in range(k)])

        if k == 1:
            sk[k - 1] = 0
        else:
            sk[k - 1] = np.std(np.log(W_kb[:, k - 1]))

    Gap_k = np.mean(np.log(W_kb), axis=0) - np.log(W_k)

    for k in range(1, K_max):
        if Gap_k[k - 1] >= Gap_k[k] - sk[k]:
            return k

    return K_max
