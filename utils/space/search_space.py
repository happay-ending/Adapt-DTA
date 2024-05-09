#!/usr/bin/python3
# -*- coding: utf-8 -*-

import typing as _typ

import torch
import torch.nn.functional as F

# from autogl.module.nas.backend import *
# from autogl.module.nas.space import BaseSpace
# from autogl.module.nas.space.operation import act_map
# from autogl.module.nas.space.operation_pyg import gnn_map
from nni.nas.pytorch import mutables
from torch import nn

from utils.model.base import BaseAutoModel
from utils.model.nas_utils import bk_feat, bk_gconv
from utils.space.base import BaseSpace
from utils.space.space_util import act_map, gnn_map, pooling_map

GRAPHNAS_DEFAULT_GNN_OPS = [
    "gat_mean",  # GAT with 1 heads
    "gat_max",
    "gat_sum",
    "gcn_mean",  # GCN
    "gcn_max",  # GCN
    "gcn_sum",  # GCN
    "cheb_mean",  # chebnet
    "cheb_max",  # chebnet
    "cheb_sum",  # chebnet
    "gin_mean",  # ginnet
    "gin_max",  # ginnet
    "gin_sum",  # ginnet gin在DTA中有用
    "sage_mean",  # sage
    "sage_max",  # sage
    "sage_sum",  # sage
    # "arma_mean",
    # "arma_max",
    # "arma_sum",
    "sg_mean",  # simplifying gcn
    "sg_max",  # simplifying gcn
    "sg_sum",  # simplifying gcn
    "linear_none",  # skip connection
    "zero_none",  # skip connection
]


GRAPHNAS_DEFAULT_ACT_OPS = [
    # "sigmoid", "tanh", "relu", "linear",
    #  "softplus", "leaky_relu", "relu6", "elu"
    "sigmoid",
    "tanh",
    "relu",
    "elu",
    "leaky_relu",
]

GRAPHNAS_DEFAULT_CON_OPS = ["add", "product", "concat"]



# GRAPHNAS_DEFAULT_READOUT_OPS = ["sum","mean","max","set2set","attention"]
GRAPHNAS_DEFAULT_READOUT_OPS = ["sum", "mean", "max", "attention"]


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.lambd)


class StrModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.str = lambd

    def forward(self, *args, **kwargs):
        return self.str

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.str)


def act_map_nn(act):
    return LambdaModule(act_map(act))


# def pool_map_nn(graph_pooling,emb_dim,processing_steps=2):
#     return LambdaModule(pooling_map(graph_pooling,emb_dim,processing_steps))


def map_nn(l):
    return [StrModule(x) for x in l]


class ArchitectureSpace(BaseSpace):
    def __init__(
            self,
            hidden_dim: _typ.Optional[int] = 64,
            layer_number: _typ.Optional[int] = 2,
            dropout: _typ.Optional[float] = 0.5,
            mol_input_dim: _typ.Optional[int] = None,
            prt_input_dim: _typ.Optional[int] = None,
            output_dim: _typ.Optional[int] = None,
            gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_GNN_OPS,
            act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
            con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_CON_OPS,
            pool_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_READOUT_OPS
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.mol_input_dim = mol_input_dim
        self.prt_input_dim = prt_input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.con_ops = con_ops
        self.pool_ops = pool_ops
        self.dropout = dropout

    def instantiate(
            self,
            hidden_dim: _typ.Optional[int] = None,
            layer_number: _typ.Optional[int] = None,
            dropout: _typ.Optional[float] = None,
            mol_input_dim: _typ.Optional[int] = None,
            prt_input_dim: _typ.Optional[int] = None,
            output_dim: _typ.Optional[int] = None,
            gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
            act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
            con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
            pool_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = None,
    ):
        # global layer
        super().instantiate()
        self.dropout = dropout or self.dropout
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.layer_number = layer_number or self.layer_number
        self.mol_input_dim = mol_input_dim or self.mol_input_dim
        self.prt_input_dim = prt_input_dim or self.prt_input_dim
        self.output_dim = output_dim or self.output_dim
        self.gnn_ops = gnn_ops or self.gnn_ops
        self.act_ops = act_ops or self.act_ops
        self.con_ops = con_ops or self.con_ops
        self.pool_ops = pool_ops or self.pool_ops
        # print("self.mol_input_dim:",self.mol_input_dim)
        # print("self.prt_input_dim:",self.prt_input_dim)
        self.mol_preproc0 = nn.Linear(self.mol_input_dim, self.hidden_dim)
        self.mol_preproc1 = nn.Linear(self.mol_input_dim, self.hidden_dim)
        self.prt_preproc0 = nn.Linear(self.prt_input_dim, self.hidden_dim)
        self.prt_preproc1 = nn.Linear(self.prt_input_dim, self.hidden_dim)
        self.norm = nn.BatchNorm1d(self.hidden_dim)
        mol_node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        prt_node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for layer in range(2, self.layer_number + 2):
            mol_node_labels.append(f"mol_op_{layer}")
            prt_node_labels.append(f"prt_op_{layer}")
            # print(f"mol_op_{layer}",f"prt_op_{layer}")
            setattr(
                self,
                f"mol_in_{layer}",
                self.setInputChoice(
                    layer,
                    choose_from=mol_node_labels[:-1],
                    n_chosen=1,
                    return_mask=False,
                    key=f"mol_in_{layer}",
                ),
            )
            # setattr(
            #     self,
            #     f"mol_aggr_{layer}",
            #     self.setInputChoice(
            #         layer,
            #         choose_from=self.gnn_aggr,
            #         n_chosen=1,
            #         return_mask=False,
            #         key=f"mol_aggr_{layer}",
            #     ),
            # )
            setattr(
                self,
                f"mol_op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        gnn_map(op, self.hidden_dim, self.hidden_dim)
                        for op in self.gnn_ops
                    ],
                    key=f"mol_op_{layer}",
                ),
            )

            setattr(
                self,
                f"prt_in_{layer}",
                self.setInputChoice(
                    layer,
                    choose_from=prt_node_labels[:-1],
                    n_chosen=1,
                    return_mask=False,
                    key=f"prt_in_{layer}",
                ),
            )

            # setattr(
            #     self,
            #     f"prt_aggr_{layer}",
            #     self.setInputChoice(
            #         layer,
            #         choose_from=self.gnn_aggr,
            #         n_chosen=1,
            #         return_mask=False,
            #         key=f"prt_aggr_{layer}",
            #     ),
            # )

            setattr(
                self,
                f"prt_op_{layer}",
                self.setLayerChoice(
                    layer,
                    [
                        gnn_map(op, self.hidden_dim, self.hidden_dim)
                        for op in self.gnn_ops
                    ],
                    key=f"prt_op_{layer}",
                ),
            )

            # setattr(
            #     self,
            #     f"prt_act_{layer}",
            #     self.setLayerChoice(
            #         layer, [act_map_nn(a) for a in self.act_ops], key=f"prt_act_{layer}"
            #     ),
            # )
        # print("选择各自的concat操作。")skip操作
        if len(self.con_ops) > 1:
            setattr(
                self,
                "mol_concat",
                self.setLayerChoice(
                    2 * layer, map_nn(self.con_ops), key="mol_concat"
                ),
            )
            setattr(
                self,
                "prt_concat",
                self.setLayerChoice(
                    2 * layer, map_nn(self.con_ops), key="prt_concat"
                ),
            )
        # print("选择各自的激活函数。")
        setattr(
            self,
            "mol_act",
            self.setLayerChoice(
                2 * layer + 1, [act_map_nn(a) for a in self.act_ops], key="mol_act"
            ),
        )
        setattr(
            self,
            "prt_act",
            self.setLayerChoice(
                2 * layer + 1, [act_map_nn(a) for a in self.act_ops], key="prt_act"
            ),
        )

        # 用于将按层拼接的矩阵维度和add/product统一到hidden_dim维度
        self.mol_reduce = nn.Linear(self.hidden_dim * self.layer_number, self.hidden_dim)
        self.prt_reduce = nn.Linear(self.hidden_dim * self.layer_number, self.hidden_dim)

        setattr(
            self,
            "mol_pool",
            self.setLayerChoice(
                2 * layer + 2, [pooling_map(a, self.hidden_dim) for a in self.pool_ops], key="mol_pool"
            ),
        )
        setattr(
            self,
            "prt_pool",
            self.setLayerChoice(
                2 * layer + 2, [pooling_map(a, self.hidden_dim) for a in self.pool_ops], key="prt_pool"
            ),
        )

        self._initialized = True
        # self.classifier1 = nn.Linear(
        #     self.hidden_dim * self.layer_number, self.output_dim
        # )
        # self.classifier2 = nn.Linear(self.hidden_dim, self.output_dim)
        # self.classifier = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.hidden_dim * 2, self.output_dim), nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

        self.reset_parameters()

    # added 2023-11-28
    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def forward(self, mol, prt):
        mol_x, mol_batch = mol.x, mol.batch
        prt_x, prt_batch = prt.x, prt.batch
        # mol_x = bk_feat(mol)
        # print('size')
        # print('mol_x:', mol_x.size(),  'batch:', mol_batch.size())
        # print('prt_x:', prt_x.size(),  'batch:', prt_batch.size())

        # mol_x = F.dropout(mol_x, p=self.dropout, training=self.training)
        # mol_pprev_, mol_prev_ = F.relu(self.mol_preproc0(mol_x)), F.relu(self.mol_preproc1(mol_x))
        mol_pprev_, mol_prev_ = self.mol_preproc0(mol_x), self.mol_preproc1(mol_x)
        mol_prev_nodes_out = [mol_pprev_, mol_prev_]

        # prt_x = F.dropout(prt_x, p=self.dropout, training=self.training)
        # prt_pprev_, prt_prev_ = F.relu(self.prt_preproc0(prt_x)), F.relu(self.prt_preproc1(prt_x))
        prt_pprev_, prt_prev_ = self.prt_preproc0(prt_x), self.prt_preproc1(prt_x)
        prt_prev_nodes_out = [prt_pprev_, prt_prev_]

        for layer in range(2, self.layer_number + 2):
            mol_node_in = getattr(self, f"mol_in_{layer}")(mol_prev_nodes_out)
            mol_op = getattr(self, f"mol_op_{layer}")
            mol_node_out = bk_gconv(mol_op, mol, mol_node_in)
            # mol_act = getattr(self, f"mol_act_{layer}")
            # mol_node_out = mol_act(mol_node_out)
            mol_node_out = self.norm(mol_node_out)
            mol_node_out = F.relu(mol_node_out)  # activation functions
            # print("type",type(mol_node_out))
            # mol_node_out = F.dropout(mol_node_out, p=self.dropout)
            mol_prev_nodes_out.append(mol_node_out)

            prt_node_in = getattr(self, f"prt_in_{layer}")(prt_prev_nodes_out)
            prt_op = getattr(self, f"prt_op_{layer}")
            prt_node_out = bk_gconv(prt_op, prt, prt_node_in)
            # prt_act = getattr(self, f"prt_act_{layer}")
            # prt_node_out = prt_act(prt_node_out)
            prt_node_out = self.norm(prt_node_out)
            prt_node_out = F.relu(prt_node_out)  # activation functions
            # prt_node_out = F.dropout(prt_node_out, p=self.dropout)
            prt_prev_nodes_out.append(prt_node_out)
            # print("第{}卷积层.".format(layer))
            # print("卷积层mol的性状：", mol_node_out.size())
            # print("卷积层prt的性状：", prt_node_out.size())

        if len(self.con_ops) > 1:
            mol_con = getattr(self, "mol_concat")()
            prt_con = getattr(self, "prt_concat")()
            # print("mol_concat:",mol_con)
            # print("prt_concat:",prt_con)
        elif len(self.con_ops) == 1:
            mol_con = self.con_ops[0]
            prt_con = self.con_ops[0]
        else:
            mol_con = "concat"
            prt_con = "concat"

        mol_states = mol_prev_nodes_out
        prt_states = prt_prev_nodes_out
        if mol_con == "concat":
            mol_x = torch.cat(mol_states[2:], dim=1)
            mol_x = self.mol_reduce(mol_x)
        else:
            tmp = mol_states[2]
            for i in range(3, len(mol_states)):
                if mol_con == "add":
                    tmp = torch.add(tmp, mol_states[i])
                elif mol_con == "product":
                    tmp = torch.mul(tmp, mol_states[i])
            mol_x = tmp

        if prt_con == "concat":
            prt_x = torch.cat(prt_states[2:], dim=1)
            prt_x = self.prt_reduce(prt_x)
        else:
            tmp = prt_states[2]
            for i in range(3, len(prt_states)):
                if prt_con == "add":
                    tmp = torch.add(tmp, prt_states[i])
                elif prt_con == "product":
                    tmp = torch.mul(tmp, prt_states[i])
            prt_x = tmp

        mol_x = self.norm(mol_x)
        prt_x = self.norm(prt_x)

        mol_act = getattr(self, "mol_act")
        prt_act = getattr(self, "prt_act")

        mol_x = mol_act(mol_x)
        prt_x = prt_act(prt_x)

        mol_x = F.dropout(mol_x, p=self.dropout)
        prt_x = F.dropout(prt_x, p=self.dropout)
        # print("聚合前mol的性状：",mol_x.size())
        # print("聚合前prt的性状：",prt_x.size())
        mol_pool = getattr(self, "mol_pool")
        prt_pool = getattr(self, "prt_pool")

        # print("mol_pool====",mol_pool)
        # print("prt_pool====", prt_pool)

        mol_x = mol_pool(mol_x, mol_batch)
        prt_x = prt_pool(prt_x, prt_batch)

        mol_x = self.norm(mol_x)
        prt_x = self.norm(prt_x)
        # added
        mol_x = mol_act(mol_x)
        prt_x = prt_act(prt_x)
        mol_x = F.dropout(mol_x, p=self.dropout)
        prt_x = F.dropout(prt_x, p=self.dropout)

        x = torch.cat((mol_x, prt_x), dim=-1)
        return self.classifier(x)
        # if con == "concat":
        #     x = self.classifier1(x)
        # else:
        #     x = self.classifier2(x)
        # return F.log_softmax(x, dim=1)

    def parse_model(self, selection, device) -> BaseAutoModel:
        # return AutoGCN(self.input_dim, self.output_dim, device)
        return self.wrap().fix(selection)
