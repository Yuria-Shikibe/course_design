import torch
import numpy as np
import os

try:
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
except ImportError:
    sk_cosine_similarity = None


def _cosine_similarity(vectors):
    n = len(vectors)
    cs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            denom = (norm_i * norm_j) + 1e-8
            cs[i, j] = np.dot(vectors[i], vectors[j]) / denom
    return cs


class FoolsGoldAggregator:
    def __init__(self, helper):
        self.helper = helper

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        wv = self._compute_weights(weight_accumulator_by_client, sampled_participants)
        self._average_models(global_model, weight_accumulator_by_client, sampled_participants, wv)
        return True

    def _compute_weights(self, weight_accumulator_by_client, sampled_participants):
        config = self.helper.config
        num = len(sampled_participants)
        layer_name = self._get_layer_name()

        for i in sampled_participants:
            index = sampled_participants.index(i)
            self._save_history(index, weight_accumulator_by_client, userID=i)

        his = []
        folderpath = f'{config.folder_path}/foolsgold'
        for i in sampled_participants:
            history_name = f'{folderpath}/history_{i}.pth'
            if os.path.exists(history_name):
                his_i_params = torch.load(history_name, map_location='cpu')
                for name, data in his_i_params.items():
                    if layer_name in name:
                        his = np.append(his, (data.cpu().numpy()).flatten())
                        break

        if len(his) == 0:
            return np.ones(num) / num

        his = np.reshape(his, (num, -1))

        cs = _cosine_similarity(his) - np.eye(num)
        maxcs = np.max(cs, axis=1) + 1e-5
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        if np.max(wv) > 0:
            wv = wv / np.max(wv)
        wv[(wv == 1)] = 0.99

        wv = (np.log((wv / (1 - wv)) + 1e-5) + 0.5)
        wv[(np.isinf(wv)) | (wv > 1)] = 1
        wv[(wv < 0)] = 0

        return wv

    def _get_layer_name(self):
        dataset = self.helper.config['dataset']
        if dataset in ['cifar10', 'cifar100']:
            return 'linear'
        elif dataset == 'tiny-imagenet-200':
            return 'fc'
        else:
            return 'fc2'

    def _save_history(self, index, weight_accumulator_by_client, userID):
        folderpath = f'{self.helper.config.folder_path}/foolsgold'
        os.makedirs(folderpath, exist_ok=True)
        if not isinstance(weight_accumulator_by_client, list):
            return
        if index < len(weight_accumulator_by_client):
            filepath = f'{folderpath}/history_{userID}.pth'
            torch.save(weight_accumulator_by_client[index], filepath)

    def _average_models(self, global_model, weight_accumulator_by_client, sampled_participants, wv):
        from collections import OrderedDict
        lr = 1
        device = self.helper.device

        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        total_weight = np.sum(wv)
        if total_weight == 0:
            total_weight = len(sampled_participants)
            wv = np.ones(len(sampled_participants))

        for idx, i in enumerate(sampled_participants):
            index = sampled_participants.index(i)
            if index < len(weight_accumulator_by_client):
                client_weight = weight_accumulator_by_client[index]
                w = wv[idx] / total_weight
                for name, data in global_model.state_dict().items():
                    if name == 'decoder.weight':
                        continue
                    averaged_weights[name] += client_weight[name] * w

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * lr
            update_per_layer = update_per_layer.detach().clone().to(dtype=data.dtype)
            data.add_(update_per_layer.to(device))
