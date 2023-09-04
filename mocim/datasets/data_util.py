# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' 单个数据帧的类 '''

import numpy as np
import torch
import trimesh


class FrameData:
    # 关键帧的类，保存了深度图像，轨迹，法向量等
    def __init__(
        self,
        frame_id=None,
        im_batch=None,
        im_batch_np=None,
        depth_batch=None,
        depth_batch_np=None,
        T_WC_batch=None,
        T_WC_batch_np=None,
        normal_batch=None,
        # 修改：加入分块得分
        score_batch=None,
        frame_avg_losses=None,
        T_WC_track=None,
        T_WC_gt=None,
    ):
        super(FrameData, self).__init__()
        self.frame_id = frame_id
        self.im_batch = im_batch
        self.im_batch_np = im_batch_np
        self.depth_batch = depth_batch
        self.depth_batch_np = depth_batch_np
        self.T_WC_batch = T_WC_batch
        self.T_WC_batch_np = T_WC_batch_np
        self.normal_batch = normal_batch
        # 新增，用于计算每个小块的得分
        self.score_batch = score_batch
        self.frame_avg_losses = frame_avg_losses
        # for pose refinement
        self.T_WC_track = T_WC_track
        self.T_WC_gt = T_WC_gt
        self.count = 0 if frame_id is None else len(frame_id)

    def add_frame_data(self, data, replace):
        """
        Add new FrameData to existing FrameData.
        添加一个帧进来，如果上一帧是关键帧就不replace，如果上一帧是最新可视化帧replace
        """
        self.frame_id = expand_data(self.frame_id, data.frame_id, replace)
        self.im_batch = expand_data(self.im_batch, data.im_batch, replace)
        self.im_batch_np = expand_data(self.im_batch_np, data.im_batch_np, replace)
        self.depth_batch = expand_data(self.depth_batch, data.depth_batch, replace)
        self.depth_batch_np = expand_data(self.depth_batch_np, data.depth_batch_np, replace)
        self.T_WC_batch = expand_data(self.T_WC_batch, data.T_WC_batch, replace)
        self.T_WC_batch_np = expand_data(self.T_WC_batch_np, data.T_WC_batch_np, replace)
        self.normal_batch = expand_data(self.normal_batch, data.normal_batch, replace)
        # dyndyn：加入score
        self.score_batch = expand_data(self.score_batch, data.score_batch, replace)
        device = data.im_batch.device
        empty_dist = torch.zeros([1], device=device)
        self.frame_avg_losses = expand_data(self.frame_avg_losses, empty_dist, replace)
        if data.T_WC_gt is not None:
            self.T_WC_gt = expand_data(
                self.T_WC_gt, data.T_WC_gt, replace)

    def __len__(self):
        # 关键帧的长度
        return 0 if self.frame_id is None else len(self.frame_id)


def expand_data(batch, data, replace=False):
    # 将新FrameData属性添加到现有FrameData属性。
    # 串联或替换批处理中的最后一行，replace
    cat_fn = np.concatenate
    if torch.is_tensor(data):
        cat_fn = torch.cat
    # 如果没有就是初始化
    if batch is None:
        batch = data
    # 如果有数据
    else:
        if replace is False:
            # 不替换，就放在后面
            batch = cat_fn((batch, data))
        else:
            # 替换，就代替最后一个，一般是替换当前帧
            batch[-1] = data[0]
    return batch


def scene_properties(mesh_path):
    # 场景属性，没有用，在trainer里面有一个set_scene_properties
    scene_mesh = trimesh.exchange.load.load(mesh_path, process=False)
    T_extent_to_scene, bound_scene_extents = trimesh.bounds.oriented_bounds(scene_mesh)
    T_extent_to_scene = np.linalg.inv(T_extent_to_scene)
    scene_center = scene_mesh.bounds.mean(axis=0)
    return T_extent_to_scene, bound_scene_extents, scene_center


def save_trajectory(traj, file_name, format="replica", timestamps=None):
    # 保存轨迹，没有用
    traj_file = open(file_name, "w")
    if format == "replica":
        for idx, T_WC in enumerate(traj):
            time = timestamps[idx]
            traj_file.write('{} '.format(time))
            np.savetxt(traj_file, T_WC[:3, :].reshape([1, 12]), fmt="%f")
    elif format == "TUM":
        for idx, T_WC in enumerate(traj):
            quat = trimesh.transformations.quaternion_from_matrix(T_WC[:3, :3])
            quat = np.roll(quat, -1)
            trans = T_WC[:3, 3]
            time = timestamps[idx]
            traj_file.write('{} '.format(time))
            np.savetxt(traj_file, trans.reshape([1, 3]), fmt="%f", newline=" ")
            np.savetxt(traj_file, quat.reshape([1, 4]), fmt="%f",)
    traj_file.close()
