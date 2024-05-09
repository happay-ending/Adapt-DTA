#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import Type

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score, BinaryAccuracy

from utils.model.nas_utils import bk_label
from utils.space.base import BaseSpace


class CPI_Estimator():

    def __init__(self, loss_f="nll_loss", evaluation=None, optimizer_type: str = "adam",
                 lr: float = 5e-3, lr_scheduler_type: str = "steplr", weight_decay: float = 5e-4):
        # def __init__(self, loss_f="nll_loss", evaluation=None):
        #     super().__init__(loss_f, evaluation)
        self.loss_f = loss_f
        if evaluation is None:
            evaluation = ["acc"]
        evaluation_dict = {"roc_auc": BinaryAUROC(),  # score
                           "pr_auc": BinaryAveragePrecision(thresholds=None),  # score
                           "acc": BinaryAccuracy(),  # pred score
                           "f1": BinaryF1Score(),  # pred score
                           "mse": MeanSquaredError(),  # score
                           "mae": MeanAbsoluteError(),  # score
                           "r2": R2Score(),  # score
                           "pcc": PearsonCorrCoef(),  # score
                           # "scc": SpearmanCorrCoef(),# score
                           }
        evaluation_fun = {}
        for eva in evaluation:
            if eva in ["acc", "f1", "roc_auc", "pr_auc", "mse", "mae", "r2", "pcc", "scc"]:
                evaluation_fun[eva] = evaluation_dict[eva]
            else:
                raise ValueError("Currently not support metric function {}".format(eva))
        self.evaluation = evaluation_fun

        self.lr = lr
        self.weight_decay = weight_decay

        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD
        else:
            raise ValueError("Currently optimizer {} error ".format(optimizer_type))

        self.lr_scheduler_type = lr_scheduler_type

    def infer(self, model: BaseSpace, dataloader):
        device = next(model.parameters()).device
        # print("device",device)
        data_mol = dataloader[0].to(device)
        data_pro = dataloader[1].to(device)

        # dset = list(dataloader)[0]
        # mask = bk_mask(dset, mask)

        # pred = model(dset)[mask]
        pred = model(data_mol, data_pro)
        # print("pred.shape",pred.shape)
        # print("pred:",pred)
        # label = bk_label(dset)
        y = bk_label(data_mol)
        # y = label[mask]

        loss = getattr(F, self.loss_f)(pred, y)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()

        y = y.cpu()
        # metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
        metrics_dict = {eva.get_eval_name(): eva.evaluate(probs, y) for eva in self.evaluation}
        # print(metrics_dict)
        return metrics_dict, loss

    def train_valid(self, model: BaseSpace, device, train_loader, val_loader, ):
        model = model.to(device)
        # print("device", device)
        optimizer = self.optimizer(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if type(self.lr_scheduler_type) == str and self.lr_scheduler_type == "steplr":
            self.scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        elif type(self.lr_scheduler_type) == str and self.lr_scheduler_type == "multisteplr":
            self.scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif type(self.lr_scheduler_type) == str and self.lr_scheduler_type == "exponentiallr":
            self.scheduler = ExponentialLR(optimizer, gamma=0.1)
        elif type(self.lr_scheduler_type) == str and self.lr_scheduler_type == "reducelronplateau":
            self.scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10, verbose=True)
        else:
            self.scheduler = None

        # losses = []
        model.train()

        for data in train_loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            y = bk_label(data_mol)

            logits = model(data_mol, data_pro)

            if self.loss_f == "nll_loss":
                logits = F.log_softmax(logits, dim=1)
                loss = getattr(F, self.loss_f)(logits, y)
            elif self.loss_f == "binary_cross_entropy":
                one_hot_y = F.one_hot(y, 2)
                one_hot_y = torch.as_tensor(one_hot_y, dtype=torch.float32)
                logits = F.softmax(logits, dim=1)  # 分类中用
                loss = getattr(F, self.loss_f)(logits, one_hot_y)
            elif self.loss_f == "mse_loss":
                loss = getattr(F, self.loss_f)(logits, y)
            else:
                raise ValueError("Currently not support loss function {}".format(self.loss_f))

            optimizer.zero_grad()
            loss.backward()
            # print("检查model模型未使用的参数：")
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)

            optimizer.step()
            # losses.append(loss.item())

        # print("train loss:", np.average(np.array(losses)))

        for key, eval in self.evaluation.items():
            self.evaluation[key] = eval.to(device)

        losses = []
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data_mol = data[0].to(device)
                data_pro = data[1].to(device)
                y = bk_label(data_mol)

                logits = model(data_mol, data_pro)

                if self.loss_f == "nll_loss":
                    logits = F.log_softmax(logits, dim=1)
                    val_loss = getattr(F, self.loss_f)(logits, y)

                    logits = F.softmax(logits, dim=1)  # 分类中用
                    preds = logits.argmax(dim=1)
                    scores = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值
                    for key, eva in self.evaluation.items():
                        if key in ["acc", "f1"]:
                            eva.update(preds, y)
                        else:
                            eva.update(scores, y)
                elif self.loss_f == "binary_cross_entropy":
                    one_hot_y = F.one_hot(y, 2)
                    one_hot_y = torch.as_tensor(one_hot_y, dtype=torch.float32)
                    logits = F.softmax(logits, dim=1)  # 分类中用
                    val_loss = getattr(F, self.loss_f)(logits, one_hot_y)
                    preds = logits.argmax(dim=1)
                    scores = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值
                    for key, eva in self.evaluation.items():
                        if key in ["acc", "f1"]:
                            eva.update(preds, y)
                        else:
                            eva.update(scores, y)
                elif self.loss_f == "mse_loss":
                    val_loss = getattr(F, self.loss_f)(logits, y)
                    scores = logits.view(-1)  # 回归中用
                    for key, eva in self.evaluation.items():
                        eva.update(scores, y)
                else:
                    raise ValueError("Currently not support loss function {}".format(self.loss_f))

                losses.append(val_loss.item())

        metric_dict = {}
        for key, eval in self.evaluation.items():
            metric_dict[key + "_avg"] = eval.compute().item()
            eval.reset()

        metric_dict["model_params"] = model.state_dict()
        metric_dict["optim_params"] = optimizer.state_dict()

        loss_avg = np.average(np.array(losses))
        # 更新学习率
        if type(self.lr_scheduler_type) == str and self.lr_scheduler_type == "reducelronplateau":
            self.scheduler.step(loss_avg)
        else:
            self.scheduler.step()

        torch.cuda.empty_cache()

        return metric_dict, loss_avg

    def testing(self, model, device, test_loader):
        model = model.to(device)

        for key, eva in self.evaluation.items():
            self.evaluation[key] = eva.to(device)

        losses = []
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                data_mol = data[0].to(device)
                data_pro = data[1].to(device)
                y = bk_label(data_mol)

                logits = model(data_mol, data_pro)

                if self.loss_f == "nll_loss":
                    logits = F.log_softmax(logits, dim=1)
                    test_loss = getattr(F, self.loss_f)(logits, y)

                    logits = F.softmax(logits, dim=1)  # 分类中用
                    preds = logits.argmax(dim=1)
                    scores = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值
                    for key, eva in self.evaluation.items():
                        if key in ["acc", "f1"]:
                            eva.update(preds, y)
                        else:
                            eva.update(scores, y)
                elif self.loss_f == "binary_cross_entropy":
                    one_hot_y = F.one_hot(y, 2)
                    one_hot_y = torch.as_tensor(one_hot_y, dtype=torch.float32)
                    logits = F.softmax(logits, dim=1)  # 分类中用
                    test_loss = getattr(F, self.loss_f)(logits, one_hot_y)
                    preds = logits.argmax(dim=1)
                    scores = torch.select(logits, 1, 1)  # 第一个参数为索引的维度,取第1个维度中索引为1的值
                    for key, eva in self.evaluation.items():
                        if key in ["acc", "f1"]:
                            eva.update(preds, y)
                        else:
                            eva.update(scores, y)
                elif self.loss_f == "mse_loss":
                    test_loss = getattr(F, self.loss_f)(logits, y)
                    scores = logits  # 回归中用
                    for key, eva in self.evaluation.items():
                        eva.update(scores, y)
                else:
                    raise ValueError("Currently not support loss function {}".format(self.loss_f))

                losses.append(test_loss.item())

        metric_dict = {}
        for key, eva in self.evaluation.items():
            metric_dict[key + "_avg"] = eva.compute().item()
            eva.reset()

        loss_avg = np.average(np.array(losses))

        torch.cuda.empty_cache()

        return metric_dict, loss_avg
