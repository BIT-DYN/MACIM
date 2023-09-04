# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''rgb和深度图预处理的一些转换, 如gbr2rgb等'''

import cv2
import numpy as np


class BGRtoRGB(object):
    """bgr format to rgb"""
    # 将bgr的图像转成rgb
    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class DepthScale(object):
    """scale depth to meters"""
    # 把深度图像换成真实的米为单位
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale


class DepthFilter(object):
    """scale depth to meters"""
    # 和上面这个一起使用，换成深度之后再过滤一下，把太深的，转为0
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        depth[far_mask] = 0.
        return depth
