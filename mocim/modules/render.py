# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''渲染深度图像的函数，主要用来可视化'''

import torch


def sdf_render_depth(z_vals, sdf):
    """
    Basic method for rendering depth from SDF using samples along a ray.
    Assumes z_vals are ordered small -> large.
    使用沿光线的采样从SDF渲染深度的基本方法。
    假设深度的顺序为“小->大”。
    """
    # 看看每条光线采样了多少
    n = sdf.size(1) 
    # 是否是物体内部的点
    inside = sdf < 0
    ixs = torch.arange(n, 0, -1).to(sdf.device)
    # 得到内部点的索引
    mul = inside * ixs
    # 看看内部点最外面的是多少
    max_ix = mul.argmax(dim=1)
    # 这是图像数目
    arange = torch.arange(z_vals.size(0))
    # 所以深度等于内部最靠外的点的深度+它距离最近表面的值，其实不太合理
    depths = z_vals[arange, max_ix] + sdf[arange, max_ix]
    # 如果没有发现内部最靠外的点，深度设为0
    # depths[max_ix == sdf.shape[1] - 1] = 0
    return depths


def render_weighted(weights, vals, dim=-1, normalise=False):
    #使用加权和的通用渲染函数， 这就是nerf的渲染操作，没用
    """
    General rendering function using weighted sum.
    """
    weighted_vals = weights * vals
    render = weighted_vals.sum(dim=dim)
    if normalise:
        n_samples = weights.size(dim)
        render = render / n_samples
    return render
