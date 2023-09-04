# import numpy as np

# # 生成一组三维向量
# vectors = np.random.randn(100, 3)  # 100个三维向量

# # 计算旋转前的方差
# variance_before = np.var(vectors, axis=0).sum()

# # 定义旋转矩阵
# angle = np.pi / 2  # 旋转角度
# rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
#                             [np.sin(angle), np.cos(angle), 0],
#                             [0, 0, 1]])

# # 对向量进行旋转
# rotated_vectors = np.dot(vectors, rotation_matrix)

# # 计算旋转后的方差
# variance_after = np.var(rotated_vectors, axis=0).sum()

# # 比较旋转前后的方差
# print("Variance before rotation:", variance_before)
# print("Variance after rotation:", variance_after)


# import torch

# # 示例张量
# x = torch.tensor([[0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
#                   [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
#                   [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]])

# # 生成随机索引
# random_indices = torch.randperm(x.size(1))[:10]

# # 获取选择的元素的索引
# selected_indices = torch.nonzero(x)[:, 1][random_indices]

# # 构建新的张量
# selected_values = torch.zeros_like(x)
# selected_values.view(-1)[selected_indices] = 1

# print(selected_values)


# import numpy as np
# import matplotlib.pyplot as plt

# def increasing_function(x):
#     # return 0.5 * (1 - np.cos(np.pi * x))
#     threshold = 0.10
#     # 根据阈值调整自变量
#     x_adjusted = np.min(x, threshold)
#     # 计算 sigmoid 函数值
#     dis_thre = 0.5 * 1 / (1 + np.exp(-10 * (x_adjusted-threshold/2)))
#     return dis_thre

# # 生成自变量的取值范围
# x = np.linspace(0, 0.3, 100)

# # 计算因变量的取值
# y = increasing_function(x)

# # 绘制函数图像
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Increasing Function')
# plt.grid(True)
# plt.show()


# import torch

# # 假设 vectors 是一个大小为 n x 10 x 3 的张量，表示 n 组三维向量
# vectors = torch.randn((5, 10, 3))

# # 计算每组向量的方差
# variances = torch.var(vectors, dim=1)

# # 得到 n 个数字的结果
# # result = variances.flatten()

# # 打印结果
# print(vectors)
# print(variances)
# print(variances.sum(dim=-1))
# print(torch.var(vectors[0], axis=0).sum())
num = 30.65
modified_number = int(num) + 1 if int(num) % 2 == 0 else int(num)
time = [x for x in range(2, modified_number, 2)]
time.append(num)
print(time)
