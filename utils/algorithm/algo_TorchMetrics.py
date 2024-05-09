#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""==============================================
# @Project : ENAS-pytorch-master
# @File    : algo.py
# @IDE     : PyCharm
# @Author  : Austin
# @Time    : 2023/6/15 9:57
================================================"""

import os

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F

from utils.algorithm.base import BaseNAS

from utils.model.nas_utils import get_module_order, replace_layer_choice, PathSamplingLayerChoice, replace_input_choice, \
    PathSamplingInputChoice, sort_replaced_module
from utils.space.base import BaseSpace

history = []


def scale(value, last_k=10, scale_value=1.0):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


class ReinforceField:
    """
    A field with ``name``, with ``total`` choices. ``choose_one`` is true if one and only one is meant to be
    selected. Otherwise, any number of choices can be chosen.
    """

    def __init__(self, name, total, choose_one):
        self.name = name
        self.total = total
        self.choose_one = choose_one

    def __repr__(self):
        return f"ReinforceField(name={self.name}, total={self.total}, choose_one={self.choose_one})"


class ReinforceController(nn.Module):

    def __init__(
            self,
            fields,
            lstm_size=64,
            lstm_num_layers=1,
            tanh_constant=2.5,
            skip_target=0.4,
            temperature=5.0,
            entropy_reduction="sum",
            device="cpu",

    ):
        super(ReinforceController, self).__init__()
        self.fields = fields
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.skip_target = skip_target
        self.device = device

        # self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)

        # self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        # self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        # self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        # self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1,requires_grad=False)
        # self.skip_targets = nn.Parameter(
        #     torch.tensor(
        #         [1.0 - self.skip_target, self.skip_target]
        #     ),  # pylint: disable=not-callable
        #     requires_grad=False,
        # )
        assert entropy_reduction in [
            "sum",
            "mean",
        ], "Entropy reduction must be one of sum and mean."
        self.entropy_reduction = torch.sum if entropy_reduction == "sum" else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

        # self.lstm = nn.ModuleDict(
        #     {
        #         field.name: nn.LSTMCell(self.lstm_size, self.lstm_size, bias=False)
        #         for field in fields
        #     }
        # )

        # the core of controller
        self.lstm = torch.nn.LSTMCell(self.lstm_size, self.lstm_size)

        self.decoder = nn.ModuleDict(
            {
                field.name: nn.Linear(self.lstm_size, field.total)
                for field in fields
            }
        )
        self.embedding = nn.ModuleDict(
            {field.name: nn.Embedding(field.total, self.lstm_size) for field in fields}
        )

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoder:
            self.decoder[decoder].bias.data.fill_(0)

    def resample(self):
        # self.sample_log_prob = 0
        # self.sample_entropy = 0
        # self.sample_skip_penalty = 0
        log_probs = []
        entropies = []

        result = dict()

        # for j in range(2):
        #     self._initialize()
        #     for i in range(j, len(self.fields), 2):
        #         result[self.fields[i].name] = self.forward(self.fields[i])
        # self._initialize()
        # for field_index in range(len(self.fields)):
        #     result[self.fields[field_index].name] = self._sample_single(self.fields[field_index], field_index)
        # print("line 142 self.fields",self.fields)
        for j in range(2):
            inputs, hidden = self._initialize()
            for field_index in range(j, len(self.fields), 2):

                # result[field.name] = self.forward(inputs,hidden,field)
                logit, hidden = self.forward(inputs, hidden, self.fields[field_index])

                probs = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * probs).sum(1, keepdim=False)
                # print("graphnas_controller line140, entropy", entropy)
                action = probs.multinomial(num_samples=1).data  # 根据给定权重对数组进行多次采样，返回采样后的元素下标
                sampled = action[:, 0]
                selected_log_prob = log_prob.gather(
                    1, get_variable(action, self.device, requires_grad=False)) #获取action对应的概率
                # print(selected_log_prob)

                # sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
                # log_prob = self.cross_entropy_loss(logit, action[:, 0])
                # # print("log_prob",log_prob)
                #
                # log_probs.append(log_prob)
                # entropy = (
                #         log_prob * torch.exp(-log_prob)
                # )  # pylint: disable=invalid-unary-operand-type
                # # self.sample_entropy += self.entropy_reduction(entropy.detach())

                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                inputs = self.embedding[self.fields[field_index].name](sampled)

                # 打印一下log_prob，entropy，和sampled
                sampled = sampled.cpu().detach().numpy().tolist()
                # self.sample_log_prob += self.entropy_reduction(log_prob)

                if len(sampled) == 1:
                    sampled = sampled[0]
                result[self.fields[field_index].name] = sampled
        return result, torch.cat(log_probs), torch.cat(entropies)

    def _initialize(self):
        # inputs = torch.randn(1, self.lstm_size) * 0.1
        inputs = torch.zeros(1, self.lstm_size)
        hidden = (torch.zeros([1, self.lstm_size]), torch.zeros([1, self.lstm_size]))

        inputs = inputs.to(self.device)
        hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
        return inputs, hidden
        # self._inputs = torch.randn(1, self.lstm_size) * 0.1
        # self._c = [
        #     torch.zeros(
        #         (1, self.lstm_size),
        #         dtype=self._inputs.dtype,
        #         device=self._inputs.device,
        #     )
        #     for _ in range(self.lstm_num_layers)
        # ]
        # self._h = [
        #     torch.zeros(
        #         (1, self.lstm_size),
        #         dtype=self._inputs.dtype,
        #         device=self._inputs.device,
        #     )
        #     for _ in range(self.lstm_num_layers)
        # ]
        # self.sample_log_prob = 0
        # self.sample_entropy = 0
        # self.sample_skip_penalty = 0
        # self.log_probs = []
        # self.entropies = []

    # def _lstm_next_step(self):
    #     self._h, self._c = self.lstm(self._inputs, (self._h, self._c))

    def forward(self, inputs, hidden, field):
        # self._lstm_next_step()
        # self._h, self._c = self.lstm[field.name](self._inputs, (self._h, self._c))
        hx, cx = self.lstm(inputs, hidden)

        # logit = self.decoder[field.name](self._h[-1])
        logit = self.decoder[field.name](hx)
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        return logit, (hx, cx)
        # if field.choose_one:
        #     sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
        #     log_prob = self.cross_entropy_loss(logit, sampled)
        #     # print("log_prob",log_prob)
        #     self._inputs = self.embedding[field.name](sampled)
        # # else:
        # #     logit = logit.view(-1, 1)
        # #     logit = torch.cat(
        # #         [-logit, logit], 1
        # #     )  # pylint: disable=invalid-unary-operand-type
        # #     sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
        # #     skip_prob = torch.sigmoid(logit)
        # #     kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
        # #     self.sample_skip_penalty += kl
        # #     log_prob = self.cross_entropy_loss(logit, sampled)
        # #     sampled = sampled.nonzero().view(-1)
        # #     if sampled.sum().item():
        # #         self._inputs = (
        # #                 torch.sum(self.embedding[field.name](sampled.view(-1)), 0)
        # #                 / (1.0 + torch.sum(sampled))
        # #         ).unsqueeze(0)
        # #     else:
        # #         self._inputs = torch.zeros(
        # #             1, self.lstm_size, device=self.embedding[field.name].weight.device
        # #         )
        # # 打印一下log_prob，entropy，和sampled
        # sampled = sampled.cpu().detach().numpy().tolist()
        # self.sample_log_prob += self.entropy_reduction(log_prob)
        #
        # # self.log_probs.insert(index,log_prob.item())
        # self.log_probs.append(log_prob)
        # entropy = (
        #         log_prob * torch.exp(-log_prob)
        # )  # pylint: disable=invalid-unary-operand-type
        # self.sample_entropy += self.entropy_reduction(entropy.detach())
        #
        # self.entropies.append(entropy)
        # if len(sampled) == 1:
        #     sampled = sampled[0]
        # return sampled


class CPI_GraphNasRL(BaseNAS):
    def __init__(
            self,
            device="auto",
            rl_num_epochs=10,
            log_frequency=None,
            grad_clip=5.0,
            entropy_weight=0.0001,
            skip_weight=0,
            baseline_decay=0.95,
            ctrl_lr=0.00035,
            rl_steps=100,
            ctrl_kwargs=None,
            submodel_epochs=100,
            n_warmup=100,
            model_lr=5e-3,
            model_wd=5e-4,
            topk=10,
            disable_progress=False,
            hardware_metric_limit=None,
            checkpoint_dir='checkpoint',
            log=None,

    ):
        super().__init__(device)
        self.rl_num_epochs = rl_num_epochs
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.rl_steps = rl_steps
        self.grad_clip = grad_clip
        self.ctrl_kwargs = ctrl_kwargs
        self.submodel_epochs = submodel_epochs
        self.ctrl_lr = ctrl_lr
        self.n_warmup = n_warmup
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.topk_model_info = []
        self.topk = topk
        self.disable_progress = disable_progress
        self.hardware_metric_limit = hardware_metric_limit
        self.checkpoint_dir = checkpoint_dir
        self.LOGGER = log

    def search(self, space: BaseSpace, train_loader, val_loader, test_loader, cpi_estimator):
        self.model = space
        # self.dataset = dset  # .to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.estimator = cpi_estimator
        # replace choice
        self.nas_modules = []
        # print("line 326,space",space)

        k2o = get_module_order(self.model)
        # print("k2o",k2o)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
        # print("line 333 self.nas_modules",self.nas_modules)
        self.model = self.model.to(self.device)
        # print("line 335 self.model",self.model)
        # fields
        self.nas_fields = [
            ReinforceField(
                name,
                len(module),
                isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1,
            )
            for name, module in self.nas_modules
        ]
        self.controller = ReinforceController(
            fields=self.nas_fields,
            lstm_size=100,
            temperature=5.0,
            tanh_constant=2.5,
            device=self.device,
            **(self.ctrl_kwargs or {}),
        ).to(self.device)  # add  .to(self.device)

        self.ctrl_optim = torch.optim.Adam(self.controller.parameters(), lr=self.ctrl_lr)
        import torch.optim as optim
        self.ctrl_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.ctrl_optim, mode='min', factor=0.1, patience=10, verbose=True)

        # train
        with tqdm(range(self.rl_num_epochs),
                  disable=self.disable_progress) as rl_num_epoch_bar:  # self.num_epochs:强化学习次数
            for i in rl_num_epoch_bar:
                l2 = self.train_controller(i)
                rl_num_epoch_bar.set_postfix(controller_reward=l2)

        selection = self._choose_best(self.topk_model_info, self.checkpoint_dir)
        arch = space.parse_model(selection, self.device)

        return arch

    def _choose_best(self, topk_model_info, checkpoint_dir):
        '''从训练得到的top_k个模型中选择最好的一个模型'''
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        key1_results = []
        for i, model_info in enumerate(topk_model_info):
            selection = model_info[1]
            model_param = model_info[2]
            optim_param = model_info[3]

            self.arch = self.model.parse_model(selection, device=self.device)
            self.arch._model.load_state_dict(model_param)
            metric_mean, loss = self.estimator.testing(self.arch._model, self.device, self.test_loader)
            self.LOGGER.info(f"|  Top_{self.topk} model on the test set:")
            for key, value in metric_mean.items():
                self.LOGGER.info(f"|  {key:<15s} : {str(value):<23s}|")
            self.LOGGER.info(f"|  {'Loss':<15s} : {str(loss):<23s}|")
            self.LOGGER.info(f"|  {'Selection':<15s}: {selection}|")

            key1_results.append(list(metric_mean.values())[0])
            self.checkpoint_save(k=i, selection=selection, model=model_param, optimizer=optim_param,
                                 saved_dir=checkpoint_dir)

        best_model_selection = topk_model_info[np.argmax(key1_results)][1]
        best_model_param = topk_model_info[np.argmax(key1_results)][2]
        best_model_optim_param = topk_model_info[np.argmax(key1_results)][3]
        self.checkpoint_save(k='best', selection=best_model_selection, model=best_model_param,
                             optimizer=best_model_optim_param,
                             saved_dir=checkpoint_dir)
        return best_model_selection

    def train_controller(self, epoch):
        self.model.eval()
        # model = self.controller
        # optim = self.ctrl_optim
        # model.train()
        # optim.zero_grad()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        rewards = []
        losses = []
        baseline = None
        history_w_rewards = []

        # diff: graph nas train 100 and derive 100 for every epoch(10 epochs), we just train 100(20 epochs). totol num of samples are same (2000)
        # self.rl_steps：；一次强化训练中train的次数
        with tqdm(
                range(self.rl_steps), disable=self.disable_progress
        ) as rl_steps_bar:
            for ctrl_step in rl_steps_bar:
                selection, log_probs, entropies = self.controller.resample()
                # print("line 421, selection:",selection)
                self.arch = self.model.parse_model(selection, device=self.device)
                self.selection = selection
                # self._resample()

                # optim.zero_grad()

                submodel_metrics = {}
                submodel_metrics["min_loss"] = float("inf")
                early_stop = 3
                counter = 0
                loss_sum = 0
                rewards_avg = 0.0
                with tqdm(range(self.submodel_epochs), disable=self.disable_progress) as submodel_epoch_bar:
                    for submodel_epoch in submodel_epoch_bar:
                        # for submodel_epoch in range(self.submodel_epochs):
                        # 思考：用多epoch的平均性能作为reward是不是更好
                        metric, submodel_loss = self.estimator.train_valid(self.arch._model, self.device,
                                                                           self.train_loader, self.val_loader)
                        measure_key = list(metric.keys())
                        submodel_epoch_bar.set_postfix({measure_key[0]: metric[measure_key[0]], "submodel_loss": submodel_loss})
                        loss_sum += submodel_loss

                        if submodel_loss < submodel_metrics["min_loss"]:  # and train_loss < min_train_loss
                            submodel_metrics["min_loss"] = submodel_loss
                            for key, value in metric.items():
                                submodel_metrics[key] = value
                        elif submodel_loss >= loss_sum / (submodel_epoch + 1):
                            counter += 1
                            if counter > early_stop:
                                break
                        rewards_avg += metric[measure_key[0]]
                    rewards_avg /= (submodel_epoch+1)
                submodel_reward = submodel_metrics[measure_key[0]]
                rewards.append(submodel_reward)

                weighted_rewards = self.get_reward(submodel_metrics, measure_key, entropies,rewards_avg)
                history_w_rewards.append(weighted_rewards)

                # if not baseline:
                #     baseline = submodel_reward
                # else:
                #     baseline = baseline * self.baseline_decay + submodel_reward * (
                #             1 - self.baseline_decay
                #     )
                # # moving average baseline
                if baseline is None:
                    baseline = weighted_rewards
                else:
                    # baseline = np.mean(history_w_rewards[:-1],axis=0) * self.baseline_decay + weighted_rewards * (1 - self.baseline_decay)
                    baseline = baseline * self.baseline_decay + weighted_rewards * (1 - self.baseline_decay)
                #
                adv = weighted_rewards - baseline
                # print("adv:",adv)
                history.append(adv)
                adv = scale(adv, scale_value=0.5)
                adv = get_variable(adv, self.device, requires_grad=False)

                loss = -log_probs * adv
                loss = loss.sum()  # or loss.mean()

                losses.append(loss.item())

                self.ctrl_optim.zero_grad()
                loss.backward()
                self.ctrl_optim.step()
                # # 反向传播时：在求导时开启侦测
                # with torch.autograd.detect_anomaly():
                #     loss.backward()

                # print("检查controller模型未使用的参数：")
                # for name, param in self.controller.named_parameters():
                #     if param.grad is None:
                #         print(name)

                # if (ctrl_step + 1) % self.rl_steps == 0:
                #     if self.grad_clip > 0:
                #         nn.utils.clip_grad_norm_(
                #             self.controller.parameters(), self.grad_clip
                #         )

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.controller.parameters(), self.grad_clip
                    )


                rl_steps_bar.set_postfix(each_best_reward=submodel_reward, reward=max(rewards), RL_step_loss=loss.item())

                self.LOGGER.info(f"+{'-' * 70}+")
                self.LOGGER.info(f"|  {'RL_step':<20s} : {str(ctrl_step + 1):<5s} / {str(self.rl_steps):<15s}|")
                for key, value in submodel_metrics.items():
                    if "_params" not in key:
                        self.LOGGER.info(f"|  {key:<15s} : {str(value):<23s}|")
                self.LOGGER.info(f"|  {'RL_step_loss':<15s} : {str(loss.item()):<23s}|")
                self.LOGGER.info(f"|  {'Selection':<15s} : {self.selection}|")
                self.LOGGER.info(f"+{'-' * 70}+")

            if not os.path.exists(os.path.join(self.checkpoint_dir, "val")):
                os.makedirs(os.path.join(self.checkpoint_dir, "val"))
            for index, m_info in enumerate(self.topk_model_info):
                self.checkpoint_save(k=index, selection=m_info[1], model=m_info[2], optimizer=m_info[3],
                                     saved_dir=os.path.join(self.checkpoint_dir, "val"))

        losses = sum(losses) / len(losses)
        # 更新学习率
        self.ctrl_scheduler.step(losses)
        self.LOGGER.info(
            f"|  {'Architecture epoch'} : {str(epoch + 1):<5s} / {str(self.rl_num_epochs):<5s} , {'mean rewards'} : {sum(rewards) / len(rewards)} , {'mean loss'} : {losses}|")

        torch.cuda.empty_cache()

        return sum(rewards) / len(rewards)

    def _resample(self):
        selection, log_probs, entropies = self.controller.resample()
        # print("\nalgo.py 413 line, result =",result)
        self.arch = self.model.parse_model(selection, device=self.device)
        self.selection = selection

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()

    # def _infer(self, model, device, test_loader):
    #     model = model.to(device)
    #     model.eval()
    #     metrics = []
    #     losses = []
    #     for data in test_loader:
    #         metric, loss = self.estimator.infer(model, data)
    #         metrics.append(np.array(list(metric.values())))
    #         losses.append(loss.item())
    #
    #     mean_name = list(map(lambda x: x + "_avg", metric))
    #     std_name = list(map(lambda x: x + "_std", metric))
    #
    #     metric_avg = np.average(np.array(metrics), 0)
    #     metrix_std = np.std(np.array(metrics), 0)
    #     loss_mean = np.average(np.array(losses))
    #
    #     avg_metric_dict = dict(zip(mean_name, metric_avg))
    #     std_metric_dict = dict(zip(std_name, metrix_std))
    #
    #
    #     return avg_metric_dict, std_metric_dict, loss_mean
    # metric, loss = self.estimator.infer(model, val_loader)
    # return metric[0], loss, metric[1:]

    # def _infer(self, mask="train"):
    #     if mask == "train":
    #         metric, loss = self.estimator.infer(self.arch._model, self.train_loader)
    #         return metric[0], loss, metric[1:]
    #     elif mask == "val":
    #         metric, loss = self.estimator.infer(self.arch._model, self.test_loader)
    #         return metric[0], loss, metric[1:]
    #     else:
    #         print("数据集mask 值不在可选（train,val）范围内！")
    #         return "error"

    # def _train_valid(self, model, device, train_loader, test_loader, optimizer):
    #     # print(next(model.parameters()).device)
    #     model = model.to(device)
    #     model.train()
    #     # loss_all = 0
    #     for data in train_loader:
    #         optimizer.zero_grad()
    #         metric, loss = self.estimator.infer(model, data)
    #         loss.backward()
    #         optimizer.step()
    #         # loss_all += data[0].num_graphs * loss.item()
    #
    #         # return metric[0], loss, metric[1:]
    #
    #     model.eval()
    #     metrics = []
    #     losses = []
    #     for data in test_loader:
    #         metric, loss = self.estimator.infer(model, data)
    #         metrics.append(np.array(list(metric.values())))
    #         losses.append(loss.item())
    #     metric_avg = np.average(np.array(metrics), 0)
    #     loss_avg = np.average(np.array(losses))
    #     key_name = list(map(lambda x:x+"_avg",metric))
    #     # print("name_dict",key_name)
    #     avg_metric_dict = dict(zip(key_name, metric_avg))
    #     # return metric_all[0], loss_all, metric_all[1:]
    #     return avg_metric_dict,loss_avg

    def checkpoint_save(self, k, selection, model, optimizer, saved_dir):
        f = os.path.join(saved_dir, f'submodel_{k}.pth')
        checkpoint = {
            'selection': selection,
            'net': model,
            'optimizer': optimizer
        }
        torch.save(checkpoint, f)

    def get_reward(self, submodel_metrics, measure_key, entropies,rewards_avg):

        # reward = submodel_metrics[measure_key[0]]
        reward = rewards_avg


        self.topk_model_info.append(
            [submodel_metrics[measure_key[0]], self.selection, submodel_metrics[measure_key[-2]],
             submodel_metrics[measure_key[-1]]])

        self.topk_model_info.sort(key=lambda x: x[0], reverse=True)  # 默认reverse=False升序排序
        if len(self.topk_model_info) > self.topk:
            self.topk_model_info.pop()

        # first_column = list(map(lambda x: x[0], self.topk_model_info))
        if self.entropy_weight:
            np_entropies = entropies.data.cpu().numpy()
            weighted_rewards = reward + self.entropy_weight * np_entropies  # reward应是前期得到奖励的平均值

            return weighted_rewards
        else:
            return reward

def get_variable(inputs, device, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    # if cuda:
    #     out = Variable(inputs.cuda(), **kwargs)
    # else:
    #     out = Variable(inputs, **kwargs)
    return out


# def get_variable(inputs, cuda=False, **kwargs):
#     if type(inputs) in [list, np.ndarray]:
#         inputs = torch.Tensor(inputs)
#     if cuda:
#         out = Variable(inputs.cuda(), **kwargs)
#     else:
#         out = Variable(inputs, **kwargs)
#     return out
