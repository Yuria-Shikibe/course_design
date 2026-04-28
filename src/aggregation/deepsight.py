import torch
import numpy as np
import copy
from sklearn.cluster import DBSCAN


class DeepSightAggregator:
    def __init__(self, helper):
        self.helper = helper

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        chosen = self._select_clients(global_model, client_models, sampled_participants)
        print(f"DeepSight init_ids:{sampled_participants} chosen_ids:{chosen}")
        self._average_models(global_model, weight_accumulator_by_client, chosen, sampled_participants)
        return True

    def _average_models(self, global_model, weight_accumulator_by_client, chosen, sampled_participants):
        from collections import OrderedDict
        lr = 1

        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        for i in chosen:
            index = sampled_participants.index(i)
            client_weight = weight_accumulator_by_client[index]
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                averaged_weights[name] += client_weight[name]

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * (1 / len(chosen)) * lr
            update_per_layer = update_per_layer.detach().clone().to(dtype=data.dtype)
            data.add_(update_per_layer.to(self.helper.device))

    def _select_clients(self, global_model, clients, chosen_ids):
        def sanitize(data, name):
            if not np.isfinite(data).all():
                print(f"DeepSight Warning: {name} contains NaN/Inf. Sanitizing...")
                return np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
            return data

        device = self.helper.device
        global_weight = list(global_model.state_dict().values())[-2]
        global_bias = list(global_model.state_dict().values())[-1]

        biases = [(list(clients[i].state_dict().values())[-1] - global_bias) for i in chosen_ids]
        weights = [list(clients[i].state_dict().values())[-2] for i in chosen_ids]

        neups = []
        n_exceeds = []
        sC_nn2 = torch.tensor(0.0, device=device)

        for i in chosen_ids:
            idx = chosen_ids.index(i)
            C_nn = torch.sum(weights[idx] - global_weight, dim=[1]) + biases[idx] - global_bias
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2

            C_max = torch.max(C_nn2).item()
            threshold = max(0.01, 1 / len(biases)) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)

        sC_nn2_val = sC_nn2.sum()
        if sC_nn2_val > 0:
            neups = np.array([(neup / sC_nn2_val).cpu().numpy() for neup in neups])
        else:
            neups = np.array([neup.cpu().numpy() for neup in neups])

        # Compute DDIF
        dataset = self.helper.config["dataset"]
        if dataset == 'mnist':
            rand_input = torch.randn((20, 1, 28, 28)).to(device)
        elif dataset == 'tiny-imagenet-200':
            rand_input = torch.randn((64, 3, 224, 224)).to(device)
        else:
            rand_input = torch.randn((256, 3, 32, 32)).to(device)

        global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
        global_ddif = global_ddif + 1e-8
        client_ddifs = [
            torch.mean(torch.softmax(clients[i](rand_input), dim=1), dim=0) / global_ddif
            for i in chosen_ids
        ]
        client_ddifs = np.array([cd.cpu().detach().numpy() for cd in client_ddifs])

        biases_np = np.array([bias.cpu().numpy() for bias in biases])

        # DBSCAN ensemble clustering
        biases_np = sanitize(biases_np, "biases")
        cosine_labels = DBSCAN(min_samples=3, metric='cosine').fit(biases_np).labels_

        neups = sanitize(neups, "neups")
        neup_labels = DBSCAN(min_samples=3).fit(neups).labels_

        client_ddifs = sanitize(client_ddifs, "ddifs")
        ddif_labels = DBSCAN(min_samples=3).fit(client_ddifs).labels_

        N = len(neups)
        dists_from_cluster = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                dists_from_cluster[i, j] = (
                    int(cosine_labels[i] == cosine_labels[j]) +
                    int(neup_labels[i] == neup_labels[j]) +
                    int(ddif_labels[i] == ddif_labels[j])
                ) / 3.0
                dists_from_cluster[j, i] = dists_from_cluster[i, j]

        ensembles = DBSCAN(min_samples=3, metric='precomputed').fit(dists_from_cluster).labels_

        # Identify malicious clusters using n_exceed
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]

        cluster_ids = np.unique(ensembles)
        deleted_cluster_ids = []
        for cluster_id in cluster_ids:
            n_mal = sum(1 for im, c in zip(identified_mals, ensembles)
                       if c == cluster_id and im)
            cluster_size = np.sum(ensembles == cluster_id)
            if cluster_size > 0 and (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)

        final_chosen = copy.deepcopy(chosen_ids)
        for i in range(len(chosen_ids) - 1, -1, -1):
            if ensembles[i] in deleted_cluster_ids:
                del final_chosen[i]

        if len(final_chosen) == 0:
            final_chosen = copy.deepcopy(chosen_ids)

        return final_chosen
