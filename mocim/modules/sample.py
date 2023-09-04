# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''采样的一些函数  包括采样像素  采样光线  采样点'''

import torch

from mocim.geometry import transform
from mocim.eval.metrics import start_timing, end_timing


def sample_pixels(
    n_rays, n_frames, h, w, device
):
    # 采样像素，总共需要多少
    total_rays = n_rays * n_frames
    indices_h = torch.randint(0, h, (total_rays,), device=device)
    indices_w = torch.randint(0, w, (total_rays,), device=device)
    #  0，1，2，然后赋值n_rays份，这样每个图像上的采样个数是一样的
    indices_b = torch.arange(n_frames, device=device)
    indices_b = indices_b.repeat_interleave(n_rays)
    # print(indices_b)
    # print(indices_h)
    return indices_b, indices_h, indices_w

def sample_pixels_score(
    score_batch, n_rays, n_frames, h, w, h_b, w_b, factor, c, r, device
):
    # dyndyn：得到每个帧上面的每个块的采样个数
    num_points_block = (n_rays * score_batch).int()
    # 每个block至少1个，至多5个
    max = round(n_rays/(factor*factor)+4)
    min = 1
    num_points_block[num_points_block>max]=max
    num_points_block[num_points_block<min]=min

    i_w = c.repeat((n_frames,1,1))
    i_h = r.repeat((n_frames,1,1))
    indices_b = torch.arange(n_frames, device=device)
    indices_b = indices_b.repeat_interleave(factor *factor *max)
    indices_b = indices_b.reshape((-1,max))
    indices_w = torch.randint(0, w_b, (max,),device = device)
    indices_w = indices_w.repeat((n_frames, factor, factor, 1))
    indices_h = torch.randint(0, h_b, (max,),device = device)
    indices_h = indices_h.repeat((n_frames, factor, factor, 1))
    indices_w = indices_w + i_w[..., None]
    indices_w = indices_w.reshape((-1, max))
    indices_h = indices_h + i_h[..., None]
    indices_h = indices_h.reshape((-1, max))
    num_points_block = num_points_block.reshape((-1))
    available = torch.ones((n_frames*factor*factor,max))
    available[num_points_block[:,None]<=torch.arange(available.shape[1],device = device)] = 0
    available = available == 1
    indices_b = indices_b[available]
    indices_h = indices_h[available]
    indices_w = indices_w[available]
    return indices_b, indices_h, indices_w


def get_batch_data(
    depth_batch,
    T_WC_batch,
    dirs_C,
    indices_b,
    indices_h,
    indices_w,
    norm_batch=None,
    get_masks=False,
):
    # 获取采样像素的深度、光线方向和姿势。仅在深度有效的情况下渲染。
    # 获得具体的采样点的深度，展开为行向量
    depth_sample = depth_batch[indices_b, indices_h, indices_w].view(-1)
    mask_valid_depth = depth_sample != 0
    # 默认没有利用法线求sdf
    norm_sample = None
    if norm_batch is not None:
        # 需要计算法线
        norm_sample = norm_batch[indices_b, indices_h, indices_w, :].view(-1, 3)
        mask_invalid_norm = torch.isnan(norm_sample[..., 0])
        mask_valid_depth = torch.logical_and(mask_valid_depth, ~mask_invalid_norm)
        norm_sample = norm_sample[mask_valid_depth]
    # 得到深度不是0的那些深度值
    depth_sample = depth_sample[mask_valid_depth]
    # 深度不是0的像素的索引
    indices_b = indices_b[mask_valid_depth]
    indices_h = indices_h[mask_valid_depth]
    indices_w = indices_w[mask_valid_depth]

    # 这些图像的相机轨迹
    T_WC_sample = T_WC_batch[indices_b]
    # 这些采样点的射线方向
    dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(-1, 3)
    masks = None
    if get_masks:
        # mask就是看看这次测试有哪些像素点被使用了
        masks = torch.zeros(depth_batch.shape, device=depth_batch.device)
        masks[indices_b, indices_h, indices_w] = 1
    return (
        dirs_C_sample,
        depth_sample,
        norm_sample,
        T_WC_sample,
        masks,
        indices_b,
        indices_h,
        indices_w
    )


def stratified_sample(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.
    均匀采样，最小深度和最大深度之间的随机样本，每个层次内有一个样品。
    """
    if n_stratified_samples is not None: 
        # 光线分层数目
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            # 所以还是采样了19个点
            bin_limits = torch.linspace(
                0, 1, n_bins + 1,
                device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            bin_limits = torch.linspace(
                min_depth,
                max_depth,
                n_bins + 1,
                device=device,
            )[None, :]
            bin_length = (max_depth - min_depth) / (n_bins)
    elif bin_length is not None:  
        # 如果没有指定采样数，使用固定长度的的bin
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1
    # 在每个小段的位置是随机的
    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments
    # 返回采集点的深度
    return z_vals


def sample_along_rays(
    T_WC,
    min_depth,
    max_depth,
    n_stratified_samples,
    n_surf_samples,
    dirs_C,
    gt_depth=None,
    grad=False,
):
    # 在光线上采样
    with torch.set_grad_enabled(grad):
        # 将光线转换到世界坐标系
        origins, dirs_W = transform.origin_dirs_W(T_WC, dirs_C)
        # 世界坐标系光线的原点
        origins = origins.view(-1, 3)
        # 世界坐标系光线的方向
        dirs_W = dirs_W.view(-1, 3)
        # 光线的个数
        n_rays = dirs_W.shape[0]
        # 沿射线均匀分层采样  n_stratified_samples=19
        z_vals = stratified_sample(
            min_depth, max_depth,
            n_rays, T_WC.device,
            n_stratified_samples, bin_length=None
        )
        z_vals_surf = z_vals
        pc_surf = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
        # 在表面进行高斯采样
        if gt_depth is not None and n_surf_samples > 0:
            surface_z_vals = gt_depth
            # 先得到n_surf_samples - 1个均值0，方差0.1的正态分布的随机数，
            offsets = torch.normal(torch.zeros(gt_depth.shape[0], n_surf_samples - 1), 0.1).to(z_vals.device)
            # 均值改为准确深度
            near_surf_z_vals = gt_depth[:, None] + offsets
            # 如果超过了最小或最大范围，设置为最小或最大
            near_surf_z_vals = torch.clamp(
                near_surf_z_vals,
                torch.full(near_surf_z_vals.shape, min_depth).to(
                    z_vals.device),
                max_depth[:, None]
            )
            # 整个采样结束，包括表面点，8个高斯表面，20个均匀采样
            z_vals = torch.cat( (surface_z_vals[:, None], near_surf_z_vals, z_vals), dim=1)
            z_vals_surf = gt_depth[:,None]
            pc_surf = origins + (dirs_W * z_vals_surf)
        # 三维采样位置的点云，原点+方向*长度
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    # 返回所有光线的点云和长度本身 pc维度为[n, 27, 3]
    return pc, z_vals, pc_surf, z_vals_surf
