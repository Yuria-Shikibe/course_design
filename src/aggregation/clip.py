import torch


class ClipAggregator:
    def __init__(self, helper):
        self.helper = helper

    def aggregate(self, global_model, weight_accumulator, weight_accumulator_by_client,
                  client_models, sampled_participants, epoch):
        self._clip_updates(weight_accumulator_by_client, sampled_participants)
        self._average_models(global_model, weight_accumulator, weight_accumulator_by_client)
        return True

    def _clip_updates(self, weight_accumulator_by_client, sampled_participants):
        clip_factor = self.helper.config['clip_factor']
        device = self.helper.device

        for idx in range(len(sampled_participants)):
            client_update = weight_accumulator_by_client[idx]
            for key in client_update:
                if 'num_batches_tracked' in key:
                    continue
                update = client_update[key]
                l2_update = torch.norm(update, p=2)
                update.div_(max(1, l2_update.item() / clip_factor))

    def _average_models(self, global_model, weight_accumulator, weight_accumulator_by_client):
        lr = 1
        num_clients = self.helper.config['num_sampled_participants']

        # Rebuild accumulator after clipping
        for name, data in global_model.state_dict().items():
            weight_accumulator[name].zero_()

        for idx in range(len(weight_accumulator_by_client)):
            client_update = weight_accumulator_by_client[idx]
            for name in weight_accumulator:
                if name in client_update:
                    weight_accumulator[name] += client_update[name]

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * (1 / num_clients) * lr
            update_per_layer = update_per_layer.detach().clone().to(dtype=data.dtype)
            data.add_(update_per_layer.to(self.helper.device))
