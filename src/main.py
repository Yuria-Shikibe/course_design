import torch
import torch.nn as nn
import numpy as np
import argparse
import copy
import logging
import sys
import os
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.helper import Helper
from src.models import get_model
from src.dataset import get_dataset, partition_data, create_client_loaders, create_test_loader
from src.client import Client, WeightAccumulator
from src.aggregation import get_aggregator
from src.utils import set_seed, setup_logging, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with Aggregation Defense')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'tiny-imagenet-200'])
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18'])
    parser.add_argument('--agg-method', type=str, default='apra',
                        choices=['avg', 'clip', 'deepsight', 'foolsgold', 'rflbat', 'apra'])
    parser.add_argument('--num-clients', type=int, default=100)
    parser.add_argument('--num-sampled', type=int, default=10)
    parser.add_argument('--num-malicious', type=int, default=0)
    parser.add_argument('--attack-type', type=str, default='label_flip')
    parser.add_argument('--global-epochs', type=int, default=50)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--clip-factor', type=float, default=1.0)
    parser.add_argument('--non-iid', action='store_true', default=True)
    parser.add_argument('--iid', dest='non_iid', action='store_false')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-IID')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folder-path', type=str, default='./results')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--apra-pca-components', type=int, default=3)
    parser.add_argument('--apra-base-clip', type=float, default=1.0)
    parser.add_argument('--apra-k-init', type=float, default=5.0)
    parser.add_argument('--apra-k-decay', type=float, default=0.1)
    parser.add_argument('--apra-no-neup-ddif', action='store_true', default=False)
    return parser.parse_args()


def build_config(args):
    config = Config(
        dataset=args.dataset,
        data_dir=args.data_dir,
        model=args.model,
        agg_method=args.agg_method,
        num_clients=args.num_clients,
        num_sampled_participants=args.num_sampled,
        num_malicious_clients=args.num_malicious,
        attack_type=args.attack_type,
        global_epochs=args.global_epochs,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        clip_factor=args.clip_factor,
        folder_path=args.folder_path,
        seed=args.seed,
        apra_pca_components=args.apra_pca_components,
        apra_base_clip=args.apra_base_clip,
        apra_k_init=args.apra_k_init,
        apra_k_decay=args.apra_k_decay,
        apra_use_neup_ddif=not args.apra_no_neup_ddif,
    )

    if args.dataset == 'cifar100':
        config.num_classes = 100
    elif args.dataset == 'tiny-imagenet-200':
        config.num_classes = 200

    return config


def train_client(client, global_model, config, epoch):
    client.set_model(global_model)
    state_dict, loss, acc = client.local_train(
        epochs=config.local_epochs,
        lr=config.lr,
        momentum=config.momentum,
    )

    update = OrderedDict()
    for name, param in global_model.state_dict().items():
        if name in state_dict:
            update[name] = state_dict[name] - param.cpu()

    return update, loss, acc


def main():
    args = parse_args()
    config = build_config(args)
    set_seed(config.seed)
    logger = setup_logging(config)

    logger.info(f"Configuration: {vars(args)}")
    logger.info(f"Device: {config.device}")

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset}")
    train_set, test_set = get_dataset(config.dataset, config.data_dir)
    test_loader = create_test_loader(test_set, config.batch_size)

    num_classes = len(train_set.classes) if hasattr(train_set, 'classes') else config.num_classes
    config.num_classes = num_classes

    # Partition data
    client_indices = partition_data(train_set, config.num_clients, config.seed,
                                    non_iid=args.non_iid, alpha=args.alpha)
    client_loaders = create_client_loaders(train_set, client_indices, config.batch_size)

    # Initialize helper and global model
    helper = Helper(config)
    global_model = get_model(config.model, num_classes=num_classes, dataset=config.dataset)
    helper.set_global_model(global_model)

    # Create clients
    clients = []
    malicious_ids = []
    if config.num_malicious_clients > 0:
        rng = np.random.RandomState(config.seed)
        malicious_ids = rng.choice(config.num_clients, config.num_malicious_clients, replace=False).tolist()

    for i in range(config.num_clients):
        loader = client_loaders.get(i)
        if loader is None or len(loader.dataset) == 0:
            continue
        client = Client(i, loader, helper, config.device)
        client.set_model(global_model)
        if i in malicious_ids:
            client.set_malicious(config.attack_type)
        clients.append(client)

    helper.set_clients(clients)
    logger.info(f"Total clients: {len(clients)}, Malicious: {len(malicious_ids)}")

    # Get aggregator
    aggregator = get_aggregator(config.agg_method, helper)
    logger.info(f"Aggregation method: {config.agg_method}")

    # Training loop
    best_acc = 0.0
    for epoch in range(config.global_epochs):
        # Sample participants
        num_sample = min(config.num_sampled_participants, len(clients))
        sampled_participants = np.random.choice(len(clients), num_sample, replace=False).tolist()
        sampled_participants.sort()

        # Local training
        weight_accumulator = WeightAccumulator(global_model)
        weight_accumulator_by_client = []
        client_models_dict = {}

        train_losses = []
        train_accs = []

        for client_id in sampled_participants:
            client = clients[client_id]
            update, loss, acc = train_client(client, global_model, config, epoch)

            weight_accumulator.add_update(update)
            weight_accumulator_by_client.append(update)
            client_models_dict[client_id] = copy.deepcopy(client.model)
            train_losses.append(loss)
            train_accs.append(acc)

            # Save updates for RFLBAT/FoolsGold
            helper.save_client_update(client_id, update)
            helper.save_foolsgold_history(client_id, update)

        # Build client_models list in order of sampled_participants
        client_models_list = [client_models_dict.get(cid, clients[cid].model)
                              for cid in sampled_participants]

        # Aggregate
        aggregator.aggregate(
            global_model=global_model,
            weight_accumulator=weight_accumulator,
            weight_accumulator_by_client=weight_accumulator_by_client,
            client_models=client_models_list,
            sampled_participants=sampled_participants,
            epoch=epoch,
        )

        # Update all client models to global model
        for client in clients:
            client.set_model(global_model)

        # Evaluate
        test_loss, test_acc = evaluate_model(global_model, test_loader, config.device)

        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_train_acc = np.mean(train_accs) if train_accs else 0

        logger.info(
            f"Epoch {epoch + 1}/{config.global_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(global_model.state_dict(), f'{config.folder_path}/best_model.pth')

    logger.info(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == '__main__':
    main()
