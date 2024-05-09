#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
===============================================
# @Project : ENAS-pytorch-master
# @File    : search_main.py
# @IDE     : PyCharm
# @Author  : Austin
# @Time    : 2023/6/13 9:38
================================================
"""
import argparse
import datetime
import os

from utils.algorithm.algo_TorchMetrics import CPI_GraphNasRL
from utils.estimator.estimator import CPI_Estimator
from utils.log import get_logger

os.environ["AUTOGL_BACKEND"] = "pyg"

# from utils.backend._dependent_backend import DependentBackend

from utils.model.nas_utils import bk_feat

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from my_dataset import mydataset
from utils.space.search_space import ArchitectureSpace


def main(args):
    # dependentBackend = DependentBackend()
    now = datetime.datetime.now()
    log_file_name = now.strftime("%Y-%m-%d_%H-%M-%S")


    log_dir = os.path.join(args.log_dir, args.dataset)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, log_file_name)

    print(f"+{'CONFIGURATION':-^86s}+")
    print(f"|  {'DATASET':<36s} : {str(args.dataset):<45s}|")
    print(f"|  {'DATASET_threshold':<36s} : {str(args.dataset_threshold):<45s}|")

    print(f"|  {'RL_NUM_EPOCHS':<36s} : {str(args.RL_NUM_EPOCHS):<45s}|")
    print(f"|  {'RL_STEPS':<36s} : {str(args.RL_STEPS):<45s}|")
    print(f"|  {'SUB_MODEL_EPOCHS':<36s} : {str(args.SUB_MODEL_EPOCHS):<45s}|")

    print(f"|  {'NUM_LAYER':<36s} : {str(args.NUM_LAYER):<45s}|")
    print(f"|  {'BATCH_SIZE':<36s} : {str(args.BATCH_SIZE):<45s}|")
    print(f"|  {'HIDDEN_DIM':<36s} : {str(args.HIDDEN_DIM):<45s}|")
    print(f"|  {'DROPOUT':<36s} : {str(args.DROPOUT):<45s}|")
    print(f"|  {'LOSS_FN':<36s} : {str(args.LOSS_FN):<45s}|")

    print(f"|  {'log_dir'.upper():<36s} : {str(log_dir):<45s}|")
    print(f"|  {'checkpoint_dir'.upper():<36s} : {str(checkpoint_dir):<45s}|")
    print(f"|  {'log_frequency'.upper():<36s} : {str(args.log_frequency):<45s}|")
    print(f"+{'-' * 86}+")

    if args.dataset == "BindingDB":
        train_data = mydataset(root=os.path.join("./data", args.dataset), name="train")
        dev_data = mydataset(root=os.path.join("./data", args.dataset), name="dev")
        test_data = mydataset(root=os.path.join("./data", args.dataset), name="test")
        train_size = len(train_data)
        valid_size = len(dev_data)
        test_size = len(test_data)

        train_dataset = random_split(train_data, [train_size, ], generator=torch.Generator().manual_seed(1))[0]
        valid_dataset = random_split(dev_data, [valid_size, ], generator=torch.Generator().manual_seed(1))[0]
        test_dataset = random_split(test_data, [test_size, ], generator=torch.Generator().manual_seed(1))[0]

    else:
        dataset = mydataset(root=os.path.join("./data", args.dataset), name=args.dataset)

        train_size = int(0.8 * len(dataset))
        valid_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - valid_size

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size],
                                                                  generator=torch.Generator().manual_seed(1))

    dataset_threshold = args.dataset_threshold

    sub_train_dataset = train_dataset[:int(len(train_dataset)*dataset_threshold)]
    sub_valid_dataset = valid_dataset[:int(len(valid_dataset) * dataset_threshold)]
    sub_test_dataset = test_dataset[:int(len(test_dataset) * dataset_threshold)]
    train_loader = DataLoader(sub_train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, pin_memory=True,
                              num_workers=8)
    val_loader = DataLoader(sub_valid_dataset, batch_size=args.BATCH_SIZE, shuffle=False, pin_memory=True,
                            num_workers=8)
    test_loader = DataLoader(sub_test_dataset, batch_size=args.BATCH_SIZE, shuffle=False, pin_memory=True,
                             num_workers=8)


    print(f"+{'-' * 86}+")
    print(f"|  {'Total number of dataset samples':<40s} : {str(train_size + valid_size + test_size):<41s}|")
    print(
        f"|  {'Training set: total number':<40s} : {str(len(train_loader.dataset)):<10s} , {'total batches':<15s} : {str(len(train_loader)):<10s}|")
    print(
        f"|  {'Validation set: total number':<40s} : {str(len(val_loader.dataset)):<10s} , {'total batches':<15s} : {str(len(val_loader)):<10s}|")
    print(
        f"|  {'Testing set: total number':<40s} : {str(len(test_loader.dataset)):<10s} , {'total batches':<15s} : {str(len(test_loader)):<10s}|")
    print(f"+{'-' * 86}+")

    data = train_loader.dataset[0]
    mol_di = bk_feat(data[0]).shape[1]
    prt_di = bk_feat(data[1]).shape[1]

    if args.LOSS_FN == "nll_loss" or args.LOSS_FN == "binary_cross_entropy":
        output_dim = 2
        evaluation = ["acc", "roc_auc", "pr_auc"]
    elif args.LOSS_FN == "mse_loss":
        output_dim = 1
        evaluation = ["pcc","r2","mse","mae"]  # 回归
    else:
        raise ValueError("Currently not support loss function {}".format(args.LOSS_FN))

    print(f"+{'-' * 86}+")
    print(f"|  {'The feature dimension of small molecules':<40s} : {str(mol_di):<41s}|")
    print(f"|  {'The feature dimension of proteins':<40s} : {str(prt_di):<41s}|")
    print(f"|  {'The output dimension of the model':<40s} : {str(output_dim):<41s}|")
    print(f"+{'-' * 86}+")

    LOGGER = get_logger(log_dir, log_file_name)
    LOGGER.info(f"+{'CONFIGURATION':-^70s}+")
    LOGGER.info(f"|  {'DATASET':<20s} : {str(args.dataset):<45s}|")
    LOGGER.info(f"|  {'DATASET_threshold':<20s} : {str(args.dataset_threshold):<45s}|")

    LOGGER.info(f"|  {'RL_NUM_EPOCHS':<20s} : {str(args.RL_NUM_EPOCHS):<45s}|")
    LOGGER.info(f"|  {'RL_STEPS':<20s} : {str(args.RL_STEPS):<45s}|")
    LOGGER.info(f"|  {'SUB_MODEL_EPOCHS':<20s} : {str(args.SUB_MODEL_EPOCHS):<45s}|")

    LOGGER.info(f"|  {'NUM_LAYER':<20s} : {str(args.NUM_LAYER):<45s}|")
    LOGGER.info(f"|  {'BATCH_SIZE':<20s} : {str(args.BATCH_SIZE):<45s}|")
    LOGGER.info(f"|  {'HIDDEN_DIM':<20s} : {str(args.HIDDEN_DIM):<45s}|")
    LOGGER.info(f"|  {'DROPOUT':<20s} : {str(args.DROPOUT):<45s}|")
    LOGGER.info(f"|  {'LOSS_FN':<20s} : {str(args.LOSS_FN):<45s}|")

    LOGGER.info(f"|  {'log_dir'.upper():<20s} : {str(log_dir):<45s}|")
    LOGGER.info(f"|  {'checkpoint_dir'.upper():<20s} : {str(checkpoint_dir):<45s}|")
    LOGGER.info(f"|  {'log_frequency'.upper():<20s} : {str(args.log_frequency):<45s}|")
    LOGGER.info(f"+{'-' * 70}+")

    # exit()
    space = ArchitectureSpace().cuda()
    space.instantiate(hidden_dim=args.HIDDEN_DIM, layer_number=args.NUM_LAYER, dropout=args.DROPOUT,
                      mol_input_dim=mol_di,
                      prt_input_dim=prt_di, output_dim=output_dim)



    estimator = CPI_Estimator(loss_f=args.LOSS_FN,evaluation=evaluation,optimizer_type='adam',lr=0.001,lr_scheduler_type='reducelronplateau')

    algo = CPI_GraphNasRL(rl_num_epochs=args.RL_NUM_EPOCHS, log_frequency=args.log_frequency, rl_steps=args.RL_STEPS,
                          submodel_epochs=args.SUB_MODEL_EPOCHS, checkpoint_dir=checkpoint_dir, log=LOGGER)
    model = algo.search(space, train_loader, val_loader, test_loader, estimator)
    print(f"+{'-' * 86}+")
    print(model)


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python search_main.py
    parser = argparse.ArgumentParser()
    # data_config
    parser.add_argument('--dataset', type=str, default="davis", help='BindingDB,davis,kiba')
    parser.add_argument('--dataset_threshold',type=float,default=0.2)
    # algorithm_config
    parser.add_argument('--RL_NUM_EPOCHS', type=int, default=1000)
    parser.add_argument('--RL_STEPS', type=int, default=100)
    parser.add_argument('--SUB_MODEL_EPOCHS', type=int, default=40)
    # model_config
    parser.add_argument('--NUM_LAYER', type=int, default=2)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--HIDDEN_DIM', type=int, default=128)
    parser.add_argument('--DROPOUT', type=float, default=0.2)
    parser.add_argument('--LOSS_FN', type=str, default='mse_loss',
                        help='nll_loss,binary_cross_entropy,mse_loss')  # binary_cross_entropy
    # log_config
    parser.add_argument('--log_dir', type=str, default='log')
    # parser.add_argument('--log_file_name', type=str, default='CPI_GraphNAS_test')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--log_frequency', type=int, default=10)

    args = parser.parse_args()

    main(args)
