# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''各种有关计算loss的函数'''

import torch

from mocim.geometry import transform

cosSim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def bounds_ray(depth_sample, z_vals, dirs_C_sample, T_WC_sample, do_grad):
    # 直接沿着光线求取bounds
    # bounds既为障碍物真实深度-采样点深度
    bounds = depth_sample[:, None] - z_vals
    # 像素点的方向
    z_to_euclidean_depth = dirs_C_sample.norm(dim=-1)
    # bounds变成了具有方向的bounds
    bounds = z_to_euclidean_depth[:, None] * bounds
    # 计算法线
    grad = None
    if do_grad:
        # 返回查看方向向量的负值，因为sdf的梯度是从小只向大的
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)
    return bounds, grad


def bounds_normal(
    depth_sample, z_vals, dirs_C_sample, norm_sample, normal_trunc_dist,
    T_WC_sample, do_grad,
):
    # 利用法线求取normal
    ray_bounds = bounds_ray(depth_sample, z_vals, dirs_C_sample)
    costheta = torch.abs(cosSim(-dirs_C_sample, norm_sample))
    # only apply correction out to truncation distance
    sub = normal_trunc_dist * (1. - costheta)
    normal_bounds = ray_bounds - sub[:, None]
    trunc_ixs = ray_bounds < normal_trunc_dist
    trunc_vals = (ray_bounds * costheta[:, None])[trunc_ixs]
    normal_bounds[trunc_ixs] = trunc_vals
    grad = None
    if do_grad:
        grad = grad_ray(T_WC_sample, dirs_C_sample, z_vals.shape[1] - 1)
    return normal_bounds, grad


def grad_ray(T_WC_sample, dirs_C_sample, n_samples):
    """ Returns the negative of the viewing direction vector """
    # 得到全局坐标的方向，返回查看方向向量的负值，因为sdf的梯度是从小只向大的
    _, dirs_W = transform.origin_dirs_W(T_WC_sample, dirs_C_sample)
    grad = - dirs_W[:, None, :].repeat(1, n_samples, 1)
    return grad


def bounds_pc(pc, z_vals, depth_sample, do_grad=True, T_WC = None, scene_file = None):
    # 利用所有采样点计算bounds
    with torch.set_grad_enabled(False):
        # 寻找所有采样点中距离每个点最近的邻居，pc原本[n, 27, 3]
        # surf_pc既只获取表面点，深度0（pc第一个点）[n, 3]
        surf_pc = pc[:, 0]
        # 所有采样点和所有表面点的距离，[n, 27, n, 3]
        diff = pc[:, :, None] - surf_pc
        # 对xyz距离求二范数，之前求得一范数诶，[n, 27, n]
        dists = diff.norm(p=2, dim=-1)
        # 求取最小范数的索引和最小值，[n, 27]
        dists, closest_ixs = dists.min(axis=-1)
        behind_surf = z_vals > depth_sample[:, None]
        dists[behind_surf] *= -1
        bounds = dists
        # pc下求取梯度也不一样了，用的到最近点的梯度
        grad = None
        if do_grad:
            # torch.Size([147, 27])
            # tensor([[  0,   0,   0,  ...,   0,   0,   0],
            #         [  1,   1,   1,  ...,   1,   1,   1],
            #         [  2,   2,   2,  ...,   2,   2,   2]
            ix1 = torch.arange(diff.shape[0])[:, None].repeat(1, diff.shape[1])
            # torch.Size([147, 27])
            # tensor([[ 0,  1,  2,  ..., 24, 25, 26],
            #         [ 0,  1,  2,  ..., 24, 25, 26],
            #         [ 0,  1,  2,  ..., 24, 25, 26]
            ix2 = torch.arange(diff.shape[1])[None, :].repeat(diff.shape[0], 1)
            # 得到每个点距离最近点的梯度
            grad = diff[ix1, ix2, closest_ixs]
            grad = grad[:, 1:]
            grad = grad / grad.norm(dim=-1)[..., None]
            # 翻转曲面后面的渐变向量
            grad[behind_surf[:, 1:]] *= -1
        # # vis gradient vector可视化梯度用的
        # import trimesh
        # import sys
        # import numpy as np
        # gt_mesh = trimesh.load(scene_file)
        # gt_mesh.visual.face_colors=[180, 180, 180, 100]
        # # zero_pc = pc[:, 6]
        # origins = T_WC[:, :3, -1]
        # surf_pc_tm = trimesh.PointCloud(surf_pc.reshape(-1, 3).cpu(), colors=[255, 0, 0])
        # #  = [0.05,0.05,0.05]
        # lines_ = torch.cat((
        #     surf_pc.reshape(-1, 3)[:, None, :],
        #     # zero_pc.reshape(-1, 3)[:, None, :]), dim=1)
        #     origins.reshape(-1, 3)[:, None, :]), dim=1)
        # paths_ = trimesh.load_path(lines_.cpu())
        # # paths_.colors = np.array([[224, 160, 158, 200]]).repeat(len(paths_.entities),axis=0)
        # paths_.colors = np.array([[255, 0, 0, 200]]).repeat(len(paths_.entities),axis=0)
        # pc_tm = trimesh.PointCloud(pc[:, 1:].reshape(-1, 3).cpu(), colors=[138,151,213])
        # closest_surf_pts = surf_pc[closest_ixs].reshape(-1, 3)
        # lines = torch.cat((
        #     closest_surf_pts[:, None, :],
        #     pc.reshape(-1, 3)[:, None, :]), dim=1)
        # paths = trimesh.load_path(lines.cpu())
        # # paths.colors = np.array([[224, 160, 158, 200]]).repeat(len(paths.entities),axis=0)
        # paths_.colors = np.array([[255, 0, 0, 200]]).repeat(len(paths_.entities),axis=0)
        # scene_1 = trimesh.Scene([paths_, surf_pc_tm, pc_tm, gt_mesh])
        # scene_1.show()
        # scene_2 = trimesh.Scene([paths, surf_pc_tm, pc_tm, gt_mesh])
        # scene_2.show()
        # sys.exit(0)

        # '''第二种可视化，球'''
        # import trimesh
        # import sys
        # import numpy as np
        # from mocim.datasets import sdf_util

        # scene_1 = trimesh.Scene()
        # scene_2 = trimesh.Scene()

        # scene_1.set_camera()
        # scene_1.camera.focal = (600.0, 600.0)
        # scene_1.camera.resolution = (1200, 680)


        # gt_mesh = trimesh.load(scene_file)
        # gt_mesh.visual.face_colors=[180, 180, 180, 100]
        # gt_mesh.visual.material.image.putalpha(30)
        # scene_1.add_geometry(gt_mesh)
        # scene_2.add_geometry(gt_mesh)
        # cmap =  sdf_util.get_colormap(sdf_range=[-2,2])
        
        # marker = trimesh.creation.camera_marker(scene_1.camera, marker_height=0.2)
        # transform = T_WC[0].cpu().numpy()
        # # print(transform)03
        # marker[0].apply_transform(transform)
        # marker[1].apply_transform(transform)
        # marker[1].colors = ((0., 1., 0., 0.8), ) * len(marker[1].entities)
        # scene_1.add_geometry(marker)
        # scene_2.add_geometry(marker)

        # origins = T_WC[:, :3, -1]
        # surf_pc_cpu = surf_pc.reshape(-1, 3).cpu()
        # # for i in range(surf_pc_cpu.shape[0]):
        # #     sphere = trimesh.primitives.Sphere(radius=0.02, center=surf_pc_cpu[i])
        # #     sphere.visual.face_colors = [0, 0, 0]
        # #     scene_1.add_geometry(sphere)
        # for i in range(surf_pc_cpu.shape[0]):
        #     sphere = trimesh.primitives.Sphere(radius=0.04, center=surf_pc_cpu[i])
        #     sphere.visual.face_colors = [130,57,50]
        #     scene_1.add_geometry(sphere)
        # for i in range(surf_pc_cpu.shape[0]):
        #     sphere = trimesh.primitives.Sphere(radius=0.04, center=surf_pc_cpu[i])
        #     sphere.visual.face_colors = [255,255,255]
        #     scene_2.add_geometry(sphere)
        # lines_ = torch.cat((surf_pc.reshape(-1, 3)[:, None, :],origins.reshape(-1, 3)[:, None, :]), dim=1)
        # paths_ = trimesh.load_path(lines_.cpu())
        # # paths_.colors = np.array([[224, 160, 158, 200]]).repeat(len(paths_.entities),axis=0)
        # paths_.colors = np.array([[127, 127, 127, 200]]).repeat(len(paths_.entities),axis=0)
        # scene_1.add_geometry(paths_)


        # pc_tm = pc[:, 1:].reshape(-1, 3).cpu()
        # for i in range(pc_tm.shape[0]):
        #     sphere = trimesh.primitives.Sphere(radius=0.04, center=pc_tm[i])
        #     sphere.visual.face_colors = [69,137,148]
        #     # sphere.visual.face_colors = [218,165,105]
        #     scene_1.add_geometry(sphere)
        #     # scene_2.add_geometry(sphere)
       

        # closest_surf_pts = surf_pc[closest_ixs].reshape(-1, 3)
        # lines = torch.cat((closest_surf_pts[:, None, :],pc.reshape(-1, 3)[:, None, :]), dim=1)
        # paths = trimesh.load_path(lines.cpu())
        # colors = np.zeros((len(paths.entities),4))
        # for i in range(len(paths.entities)):
        #     distance = torch.Tensor(paths.vertices[paths.entities[i].end_points[0]] - paths.vertices[paths.entities[i].end_points[1]]).norm(p=2,dim=-1)
        #     color = cmap.to_rgba(distance.flatten(), alpha=1., bytes=False)
        #     paths.entities[i].color = color[::-1]
        #     colors[i]=color[::-1]
        # scene_2.add_geometry(paths)

        # pc_tm = pc[:, 1:].reshape(-1, 3).cpu()
        # sur = surf_pc[closest_ixs][:, 1:].reshape(-1, 3)
        # yes = pc[:, 1:].reshape(-1, 3)
        # for i in range(pc_tm.shape[0]):
        #     distance = (sur[i] - yes[i]).norm(p=2,dim=-1)
        #     color = cmap.to_rgba(distance.cpu().flatten(), alpha=1., bytes=False)
        #     sphere = trimesh.primitives.Sphere(radius=0.04, center=pc_tm[i])
        #     sphere.visual.face_colors = color[0]
        #     sphere.visual.face_colors[3] = 0.5
        #     scene_2.add_geometry(sphere)

        # # print(bounds.shape)
        # # print(len(paths.entities))
        # # print(paths.discrete)
        # # print(paths.entities[2].end_points)
        # # print(paths.vertex_nodes.shape)
        # # print(paths.vertices.shape)

        # scene_1.show()
        # scene_2.show()
        # sys.exit(0)
    return bounds, grad


def bounds(
    method,
    dirs_C_sample,
    depth_sample,
    T_WC_sample,
    z_vals,
    pc,
    normal_trunc_dist,
    norm_sample,
    do_grad=True,
    scene_file = None
):
    """ do_grad: compute approximate gradient vector. """
    # 计算真值的sdf的bounds和法线
    assert method in ["ray", "normal", "pc", "all_pc"]
    # dirs_C_sample像素点方向，T_WC_sample相机位置
    # 第一种方法，直接沿着光线求取bounds
    if method == "ray":
        bounds, grad = bounds_ray(depth_sample, z_vals, dirs_C_sample, T_WC_sample, do_grad)
    # 第二种方法，通过法线计算bounds
    elif method == "normal":
        bounds, grad = bounds_normal(depth_sample, z_vals, dirs_C_sample,norm_sample, normal_trunc_dist, T_WC_sample, do_grad)
    # 第三种发放，通过所有采样点计算bounds
    else:
        bounds, grad = bounds_pc(pc, z_vals, depth_sample, do_grad, T_WC = T_WC_sample, scene_file = scene_file)
    # 返回对应方法的梯度
    return bounds, grad


def sdf_loss(sdf, bounds, t, loss_type="L1", no_rand_id = False, non_grids = None):
    # 计算sdf的loss，包括表面附近的和空闲区域的
    """
        params:
        sdf: predicted sdf values. 网络预测的sdf值
        bounds: upper bound on abs(sdf) sdf的上限边界
        t: truncation distance up to which the sdf value is directly supevised. 直接支持sdf值的截断距离, 判断是否属于边界点
        loss_type: L1 or L2 loss.
    """
    # 得到两种不同位置的loss，还没有区分，都进行了计算
    free_space_loss_mat, trunc_loss_mat, non_loss_mat = full_sdf_loss(sdf, bounds, no_rand_id = no_rand_id)
    # 空闲空间的部分索引
    free_space_ixs = bounds > t
    # 只有空闲部分的diff
    free_space_loss_mat[~free_space_ixs] = 0.
    # 只有表面部分的diff
    trunc_loss_mat[free_space_ixs] = 0.
    if no_rand_id:
        free_space_loss_mat[-non_grids:] = 0.
        trunc_loss_mat[-non_grids:] = 0.
        # 只有未观测部分的diff
        non_loss_mat[:-non_grids] = 0.
        # 每个点的具体的diff
        sdf_loss_mat = free_space_loss_mat + trunc_loss_mat + non_loss_mat
    else:
        sdf_loss_mat = free_space_loss_mat + trunc_loss_mat
    # 使用L1的方式根据loss计算loss，不知道为啥
    if loss_type == "L1":
        sdf_loss_mat = torch.abs(sdf_loss_mat)
    elif loss_type == "L2":
        sdf_loss_mat = torch.square(sdf_loss_mat)
    elif loss_type == "Huber":
        L2_idx =  sdf_loss_mat < 0.1
        sdf_loss_mat[L2_idx] = torch.square(sdf_loss_mat[L2_idx])
        sdf_loss_mat[~L2_idx] = torch.square(sdf_loss_mat[~L2_idx])
    else:
        raise ValueError("Must be L1 or L2 or Huber")
    # 返回得到的loss
    return sdf_loss_mat, free_space_ixs


def sdf_loss_add_points(sdf_loss_mat, loss_type="L1"):
    if loss_type == "L1":
        sdf_loss_mat = torch.abs(sdf_loss_mat)
    elif loss_type == "L2":
        sdf_loss_mat = torch.square(sdf_loss_mat)
    elif loss_type == "Huber":
        L2_idx =  sdf_loss_mat < 0.1
        sdf_loss_mat[L2_idx] = torch.square(sdf_loss_mat[L2_idx])
        sdf_loss_mat[~L2_idx] = torch.square(sdf_loss_mat[~L2_idx])
    else:
        raise ValueError("Must be L1 or L2 or Huber")
    # 返回得到的loss
    return sdf_loss_mat



def sdf_loss_add_points_near(sdf, bounds, loss_type="L1"):
    sdf_loss_mat = sdf - bounds
    if loss_type == "L1":
        sdf_loss_mat = torch.abs(sdf_loss_mat)
    elif loss_type == "L2":
        sdf_loss_mat = torch.square(sdf_loss_mat)
    elif loss_type == "Huber":
        L2_idx =  sdf_loss_mat < 0.1
        sdf_loss_mat[L2_idx] = torch.square(sdf_loss_mat[L2_idx])
        sdf_loss_mat[~L2_idx] = torch.square(sdf_loss_mat[~L2_idx])
    else:
        raise ValueError("Must be L1 or L2 or Huber")
    # 返回得到的loss
    return sdf_loss_mat


def full_sdf_loss(sdf, target_sdf, free_space_factor=5.0, no_rand_id=False):
    """
    For samples that lie in free space before truncation region:
        loss(sdf_pred, sdf_gt) =  { max(0, sdf_pred - sdf_gt), if sdf_pred >= 0
                                  { exp(-sdf_pred) - 1, if sdf_pred < 0

    For samples that lie in truncation region:
        loss(sdf_pred, sdf_gt) = sdf_pred - sdf_gt
    """
    # dyndyn: 空闲空间的计算方式
    # free_space_loss_mat = sdf - target_sdf
    free_space_loss_mat = torch.max(
        torch.nn.functional.relu(sdf - target_sdf),
        torch.exp(-free_space_factor * sdf) - 1.
    )
    # 表面空间的计算方式
    trunc_loss_mat = sdf - target_sdf
    non_loss_mat = None
    if no_rand_id:
        postive_mask = sdf>0
        non_loss_mat_postive = torch.max(
        torch.nn.functional.relu(sdf - target_sdf),
        torch.exp(-free_space_factor * (sdf - 0.1)) - 1,
        )
        non_loss_mat_postive[~postive_mask]=0
        non_loss_mat_nagetive = torch.max(
        torch.nn.functional.relu( - sdf - target_sdf),
        torch.exp(-free_space_factor * (-sdf - 0.1)) - 1,
        )
        non_loss_mat_nagetive[postive_mask]=0
        non_loss_mat = non_loss_mat_postive + non_loss_mat_nagetive
    return free_space_loss_mat, trunc_loss_mat, non_loss_mat


def tsdf_loss(sdf, target_sdf, trunc_dist):
    # Neural RGB-D Surface Reconstruction论文的做法，没用这个
    """
    tsdf loss from: https://arxiv.org/pdf/2104.04532.pdf
    SDF values in truncation region are scaled in range [0, 1].
    """
    free_space_loss_mat = sdf - torch.ones(sdf.shape, device=sdf.device)
    trunc_loss_mat = sdf - target_sdf / trunc_dist

    return free_space_loss_mat, trunc_loss_mat


def tot_loss(
    sdf_loss_mat, grad_loss_mat, eik_loss_mat,
    free_space_ixs, bounds, eik_apply_dist,
    trunc_weight, grad_weight, non_weight,eik_weight, t, 
    sdf, multi = False, no_rand_id = False, non_grids = None,
):
    # 计算总损失，把各种损失加起来
    if multi:
        sdf_loss_mat[sdf_loss_mat>2]=2
    # # 非空闲空间的部分要乘以一个系数
    # sdf_loss_mat[~free_space_ixs] *= trunc_weight
    sdf_loss_mat[:-non_grids][~free_space_ixs[:-non_grids]] *= trunc_weight
    # # 非空闲空间的部分要乘以一个系数，这个系数和它的位置有关
    # weight_num = torch.zeros_like(sdf_loss_mat)
    # weight_num[~free_space_ixs]=((1-trunc_weight)/t)*sdf[~free_space_ixs]+trunc_weight
    # sdf_loss_mat[~free_space_ixs] *= weight_num[~free_space_ixs]
    # 最初始的sdf_loss
    sdf_loss = sdf_loss_mat.mean().item()  
    losses = {"sdf": sdf_loss }
    tot_loss_mat = sdf_loss_mat
    # 计算所有点的梯度损失
    if grad_loss_mat is not None:
        # 加到总损失
        tot_loss_mat = tot_loss_mat + grad_weight * grad_loss_mat
        grad_loss = grad_loss_mat.mean().item()
        losses["grad"] =grad_loss
    # 计算eikonal方程损失
    if eik_loss_mat is not None:
        # 只针对表面外一定距离的点计算，否则为0
        eik_loss_mat[bounds < eik_apply_dist] = 0.
        # 乘以权重
        eik_loss_mat = eik_loss_mat * eik_weight
        # 加到总损失
        tot_loss_mat = tot_loss_mat + eik_loss_mat
        eik_loss = eik_loss_mat.mean().item()
        losses["eq"] = eik_loss
    # 对于loss，只需要修改最后的，因为其他地方的non_loss都是0
    if no_rand_id:
        tot_loss_mat[-non_grids:] *= non_weight
    # 总损失再取平均，得到总平均损失
    tot_loss = tot_loss_mat.mean()
    # 返回总平均损失，总损失矩阵和损失类
    return tot_loss, tot_loss_mat, losses


def approx_loss(full_loss, binary_masks, W, H, factor=8):
    # 图像分块的损失，这个最后没用factor = 8
    w_block = W // factor
    h_block = H // factor
    # 把每个图像分8块
    loss_approx = full_loss.view(-1, factor, h_block, factor, w_block)
    # 计算每个块的总损失
    loss_approx = loss_approx.sum(dim=(2, 4))
    # 这是其中哪些点是采样点
    actives = binary_masks.view(-1, factor, h_block, factor, w_block)
    # 采样点总数
    actives = actives.sum(dim=(2, 4))
    # 如果没有采样点，除数就是1，不然除以0了
    actives[actives == 0] = 1.0
    # 计算每个块的平均损失
    loss_approx = loss_approx / actives
    return loss_approx


def frame_avg(
    total_loss_mat, depth_batch, indices_b, indices_h, indices_w,
    W, H, loss_approx_factor, binary_masks,
):
    # 帧平均损失，用来看看历史关键帧的哪些loss比较高了， depth_batch [n_used_kf, 680, 1200]
    full_loss = torch.zeros(depth_batch.shape, device=depth_batch.device)
    # 沿着最后维度求和，既得到27个采样点的总损失,indices_b x indices_h x indices_w就是total_loss_mat的第一维总数
    full_loss[indices_b, indices_h, indices_w] = total_loss_mat.sum(-1).detach()
    # 按照块计算平均loss，[n_used_kf, 8, 8]
    loss_approx = approx_loss(full_loss, binary_masks, W, H, factor=loss_approx_factor)
    factor = loss_approx.shape[1]
    # 计算图像的平均损失 [n_used_kf]
    frame_sum = loss_approx.sum(dim=(1, 2))
    frame_avg_loss = frame_sum / (factor * factor)
    # 返回分块的loss，和分帧的loss
    return loss_approx, frame_avg_loss
