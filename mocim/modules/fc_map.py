# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''主要网络架构  还有计算grad的函数'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


def gradient(inputs, outputs):
    # 计算法线，用于法线loss
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    # 计算这些点的梯度，outputs是点的sdf[n,37]值，inputs是点([n,27,3])
    points_grad = grad(
        outputs=outputs, # 求导的因变量，也就是sdf
        inputs=inputs,  # 求导的自变量，也就是xyz
        grad_outputs=d_points, # outputs如果是向量，必须写，权重均为1
        create_graph=True, # 计算高阶倒导数
        retain_graph=True, # 保留计算图
        only_inputs=True,
        allow_unused=True)[0]
    return points_grad


def chunks(
    pc,
    chunk_size,
    fc_sdf_map,
    to_cpu=False,
):
    # 按照批次，计算pc点的sdf，这是绘制切片或mesh需要用到的
    # 393216=256x256x6 或者 256x256x256
    n_pts = pc.shape[0]
    # 分不同批批理，怕显存超了
    n_chunks = int(np.ceil(n_pts / chunk_size))
    alphas = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = pc[start:end, :]
        # 获得这些位置的sdf
        alpha = fc_sdf_map(chunk)
        alpha = alpha.squeeze(dim=-1)
        if to_cpu:
            alpha = alpha.cpu()
        alphas.append(alpha)
    # 把这些数据一切变成一维返回
    alphas = torch.cat(alphas, dim=-1)
    return alphas


def fc_block(in_f, out_f):
    # 全部使用softplus作为激活函数，就是一个全连接层
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.Softplus(beta=100)
    )


def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    # 把网络参数随机初始化，正态分布
    if isinstance(m, torch.nn.Linear):
        init_fn(m.weight)


class SDFMap(nn.Module):
    # 主要网络架构
    def __init__(
        self,
        positional_encoding,
        hidden_size=256,
        hidden_layers_block=1,
        scale_output=1.,
    ):
        super(SDFMap, self).__init__()
        self.scale_output = scale_output

        self.positional_encoding = positional_encoding
        embedding_size = self.positional_encoding.embedding_size

        # 输入层，全连接网络，把255全连接为256
        self.in_layer = fc_block(embedding_size, hidden_size)
        # 隐含层1，两个全连接网络
        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid1 = torch.nn.Sequential(*hidden1)

        # 把位置编码在加进来
        self.cat_layer = fc_block(
            hidden_size + embedding_size, hidden_size)

        # 隐含层2，两个全连接网络
        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid2 = torch.nn.Sequential(*hidden2)

        # 把256维度输出为sdf值
        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        # # dyndyn 
        # self.cat_layer_1 = fc_block(hidden_size + embedding_size, hidden_size)
        # self.cat_layer_2 = fc_block(hidden_size + embedding_size, hidden_size)
        # self.cat_layer_3 = fc_block(hidden_size + embedding_size, hidden_size)
        # self.out_alpha_x = torch.nn.Linear(hidden_size + embedding_size, 1)

        # 随机初始化网络参数
        self.apply(init_weights)

    def forward(self, x, noise_std=None, pe_mask=None, sdf1=None):
        # 输入进来的点云，先进行位置编码，得到255维度
        x_pe = self.positional_encoding(x)
        # 这个没用上
        if pe_mask is not None:
            x_pe = torch.mul(x_pe, pe_mask)
        # 输入层编码为256特征
        fc1 = self.in_layer(x_pe)

        # 第一个fc层，两个全连接
        fc2 = self.mid1(fc1)
        # 此时要把结果和输入255坐标链接起来
        fc2_x = torch.cat((fc2, x_pe), dim=-1)
        # 一起输入cat层，再编码为256特征
        fc3 = self.cat_layer(fc2_x)
        # 第二个fc层
        fc4 = self.mid2(fc3)
        # 输出层解码得到sdf值
        raw = self.out_alpha(fc4)
        
        # # dyndyn: 每一层都是用cat
        # cat_in_1 = torch.cat((fc1, x_pe), dim=-1)
        # cat_out_1 = self.cat_layer_1(cat_in_1)
        # cat_in_2 = torch.cat((cat_out_1, x_pe), dim=-1)
        # cat_out_2 = self.cat_layer_2(cat_in_2)
        # out_in = torch.cat((cat_out_2, x_pe), dim=-1)
        # raw = self.out_alpha_x(out_in)

        if noise_std is not None:
            # 为什么给sdf增加噪声，不懂
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        # 对输出sdf进行缩放0.14，不懂诶
        alpha = raw * self.scale_output

        # 参数维度
        # torch.Size([256, 255])
        # torch.Size([256])
        # torch.Size([256, 256])
        # torch.Size([256])
        # torch.Size([256, 256])
        # torch.Size([256])
        # torch.Size([256, 511])
        # torch.Size([256])
        # torch.Size([256, 256])
        # torch.Size([256])
        # torch.Size([256, 256])
        # torch.Size([256])
        # torch.Size([1, 256])
        # torch.Size([1])
        return alpha.squeeze(-1)
