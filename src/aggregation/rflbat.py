import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sk_cos

from src.utils import gap_statistic


class RFLBATAggregator:
    def __init__(self, helper):
        self.helper = helper

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        chosen_ids = self._select_clients(weight_accumulator_by_client, sampled_participants)
        self._average_models(global_model, weight_accumulator_by_client, chosen_ids, sampled_participants)
        return True

    def _select_clients(self, weight_accumulator_by_client, sampled_participants):
        eps1 = 10
        eps2 = 6
        config = self.helper.config
        dataset = config['dataset']
        folder_path = config['folder_path']

        dataAll = []
        for i in sampled_participants:
            index = sampled_participants.index(i)
            if index >= len(weight_accumulator_by_client):
                continue
            loaded_params = weight_accumulator_by_client[index]
            dataList = []
            for name, data in loaded_params.items():
                if any(k in name for k in ['mnist', 'linear', 'layer4.1.conv', 'fc']):
                    dataList.extend(data.cpu().numpy().flatten().tolist())
            if dataList:
                dataAll.append(dataList)

        if len(dataAll) < 2:
            return sampled_participants

        # PCA
        pca = PCA(n_components=min(2, len(dataAll) - 1))
        X_dr = pca.fit_transform(dataAll)

        # Stage 1: Euclidean distance filtering
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = sum(np.linalg.norm(X_dr[i] - X_dr[j])
                        for j in range(len(X_dr)) if i != j)
            eu_list.append(eu_sum)

        median_eu = np.median(eu_list)
        accept = [i for i, eu in enumerate(eu_list) if eu < eps1 * median_eu]
        print(f"RFLBAT: Stage 1 accepted {len(accept)}/{len(X_dr)} clients")

        if len(accept) < 2:
            return sampled_participants

        x1 = np.array([X_dr[i] for i in accept])

        # KMeans clustering
        num_clusters = gap_statistic(x1, num_sampling=5, K_max=min(10, len(accept)), n=len(x1))
        num_clusters = max(2, num_clusters)

        k_means = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10)
        predicts = k_means.fit_predict(x1)

        # Select cluster with smallest median cosine similarity
        v_med = []
        for c in range(num_clusters):
            indices_in_cluster = [j for j in range(len(predicts)) if predicts[j] == c]
            if len(indices_in_cluster) <= 1:
                v_med.append(1.0)
                continue
            cluster_data = [dataAll[accept[j]] for j in indices_in_cluster]
            v_med.append(np.median(np.average(sk_cos(cluster_data), axis=1)))

        best_cluster = int(np.argmin(v_med))
        accept = [accept[j] for j in range(len(accept)) if predicts[j] == best_cluster]

        # Stage 2: Final outlier removal
        if len(accept) < 2:
            chosen_id = [sampled_participants[i] for i in accept]
            return chosen_id

        x2 = np.array([X_dr[i] for i in accept])
        eu_list2 = []
        for i in range(len(x2)):
            eu_sum2 = sum(np.linalg.norm(x2[i] - x2[j])
                         for j in range(len(x2)) if i != j)
            eu_list2.append(eu_sum2)

        median_eu2 = np.median(eu_list2)
        accept = [accept[i] for i in range(len(accept)) if eu_list2[i] < eps2 * median_eu2]

        chosen_ids = [sampled_participants[i] for i in accept]
        print(f"RFLBAT: Final accepted {len(chosen_ids)} clients: {chosen_ids}")
        return chosen_ids

    def _average_models(self, global_model, weight_accumulator_by_client, chosen_ids, sampled_participants):
        from collections import OrderedDict
        lr = 1
        device = self.helper.device

        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        for i in chosen_ids:
            index = sampled_participants.index(i)
            if index < len(weight_accumulator_by_client):
                client_weight = weight_accumulator_by_client[index]
                for name, data in global_model.state_dict().items():
                    if name == 'decoder.weight':
                        continue
                    averaged_weights[name] += client_weight[name]

        count = len(chosen_ids) if len(chosen_ids) > 0 else len(sampled_participants)
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * (1 / count) * lr
            update_per_layer = update_per_layer.detach().clone().to(dtype=data.dtype)
            data.add_(update_per_layer.to(device))
