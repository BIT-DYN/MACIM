# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' 数据集的三个类 ReplicaDataset  ScanNetDataset  SceneCache'''

from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os


class ReplicaDataset(Dataset):
    # 整个 Replica数据集的类，把所有数据存储下来，然后一点点拿出来，方便后面模拟实时
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 noisy_depth=False,
                 col_ext=".jpg"):
        self.Ts = None
        if traj_file is not None:
            # 这里加载的真值轨迹
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext
        # 整个没用噪声的
        self.noisy_depth = noisy_depth

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        # 给出图像帧数索引，返回对应的rgb, depth, T 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        s = f"{idx:06}"  # int variable
        if self.noisy_depth:
            depth_file = os.path.join(self.root_dir, "ndepth" + s + ".png")
        else:
            depth_file = os.path.join(self.root_dir, "depth" + s + ".png")
        rgb_file = os.path.join(self.root_dir, "frame" + s + self.col_ext)
        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)
        T = None
        if self.Ts is not None:
            T = self.Ts[idx]
        sample = {"image": image, "depth": depth, "T": T}
        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])
        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])
        return sample


class ScanNetDataset(Dataset):
    # 整个ScanNet数据集的类，把所有数据存储下来，然后一点点拿出来，方便后面模拟实时
    def __init__(self,
                 root_dir,
                 traj_file,
                 rgb_transform=None,
                 depth_transform=None,
                 col_ext=None,
                 noisy_depth=None):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "frames", "color/")
        self.depth_dir = os.path.join(root_dir, "frames", "depth/")
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        depth_file = self.depth_dir + str(idx) + ".png"
        rgb_file = self.rgb_dir + str(idx) + self.col_ext
        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)
        T = None
        if self.Ts is not None:
            T = self.Ts[idx]
        sample = {"image": image, "depth": depth, "T": T}
        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])
        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])
        return sample


class SceneCache(Dataset):
    # 场景缓存类，保存五分之一的数据，这个是评估时候用的，可以用来查找历史可视化了哪些
    # 不用上面的场景类，是因为可以get_all，减少计算成本
    def __init__(self,
                 dataset_format,
                 root_dir,
                 traj_file,
                 keep_ixs=None,
                 rgb_transform=None,
                 depth_transform=None,
                 noisy_depth=False,
                 col_ext=".jpg",):
        # 数据集那类型
        self.dataset_format = dataset_format
        # 加载真值相机轨迹
        self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.samples = []
        # 需要保留的索引，是每5帧选取一个
        if keep_ixs is not None:
            keep_ixs.sort()
        self.keep_ixs = keep_ixs
        print("Loading scene cache dataset for evaluation...")
        for idx in range(self.Ts.shape[0]):
            if keep_ixs is not None:
                if idx not in keep_ixs:
                    continue
            if dataset_format == "replicaCAD":
                # 加载深度图像和rgb图像
                s = f"{idx:06}"  # int variable
                if noisy_depth:
                    depth_file = self.root_dir + "/ndepth" + s + ".png"
                else:
                    depth_file = self.root_dir + "/depth" + s + ".png"
                rgb_file = self.root_dir + "/frame" + s + col_ext
            elif dataset_format == "ScanNet":
                depth_file = root_dir + "/frames/depth/" + str(idx) + ".png"
                rgb_file = root_dir + "/frames/color/" + str(idx) + col_ext
            depth = cv2.imread(depth_file, -1)
            image = cv2.imread(rgb_file)
            # 将图像按照对应变换成为可用的数据
            if self.rgb_transform:
                image = self.rgb_transform(image)
            if self.depth_transform:
                depth = self.depth_transform(depth)
            self.samples.append((image, depth, self.Ts[idx]))
        # 得到最终所有的缓存数据
        self.samples = np.array(self.samples)
        print("Len cached dataset", len(self.samples))

    def __len__(self):
        # 长度就是五分之一的序列长度
        return len(self.samples)

    def get_all(self):
        # 获得所有缓存的数据
        idx = np.arange(self.__len__())
        image = np.concatenate(([x[None, :] for x in self.samples[idx, 0]]))
        depth = np.concatenate(([x[None, :] for x in self.samples[idx, 1]]))
        T = np.concatenate(([x[None, :] for x in self.samples[idx, 2]]))
        sample = {
            "image": image,
            "depth": depth,
            "T": T
        }
        return sample

    def __getitem__(self, idx):
        # 得到某一帧数据，和场景类的做法一样
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.keep_ixs is not None:
            idx = [x for x in idx if x in self.keep_ixs]
            idx = np.array([np.where(self.keep_ixs == x)[0][0] for x in idx])
        image = np.concatenate(([x[None, :] for x in self.samples[idx, 0]]))
        depth = np.concatenate(([x[None, :] for x in self.samples[idx, 1]]))
        T = np.concatenate(([x[None, :] for x in self.samples[idx, 2]]))
        sample = {
            "image": image,
            "depth": depth,
            "T": T
        }
        return sample
