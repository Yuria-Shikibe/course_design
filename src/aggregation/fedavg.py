import torch
from collections import OrderedDict


class FedAvgAggregator:
    def __init__(self, helper):
        self.helper = helper

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        lr = 1
        averaged_weights = OrderedDict()
        for layer, weight in global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        chosen_id = sampled_participants
        for i in chosen_id:
            index = sampled_participants.index(i)
            client_weight = weight_accumulator_by_client[index]
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                averaged_weights[name] += client_weight[name]

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = averaged_weights[name] * (1 / len(chosen_id)) * lr
            update_per_layer = update_per_layer.detach().clone().to(dtype=data.dtype)
            data.add_(update_per_layer.to(self.helper.device))

        return True
