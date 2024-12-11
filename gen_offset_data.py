import numpy as np
import cv2
import os
import glob
import random

A_path = r'E:\JWDataSets\LEVIR-CD\LEVIR-CD\test\A'
B_path = r'E:\JWDataSets\LEVIR-CD\LEVIR-CD\test\B'
lab_path = r'E:\JWDataSets\LEVIR-CD\LEVIR-CD\test\label'
# 获取目录下所有文件名
A_files = glob.glob(os.path.join(A_path, '*.png'))
B_files = glob.glob(os.path.join(B_path, '*.png'))
lab_files = glob.glob(os.path.join(lab_path, '*.png'))

offset_save_path = r"E:\JWDataSets\LEVIR-CD1\test\offset_A_512"
save_path_512_A = r'E:\JWDataSets\LEVIR-CD1\LEVIR-CD-512-OFFSET\A_src'
save_path_512_A_off = r'E:\JWDataSets\LEVIR-CD1\LEVIR-CD-512-OFFSET\A'
save_path_512_B = r'E:\JWDataSets\LEVIR-CD1\LEVIR-CD-512-OFFSET\B'
save_path_512_label = r'E:\JWDataSets\LEVIR-CD1\LEVIR-CD-512-OFFSET\label'

if not os.path.exists(offset_save_path):
    os.makedirs(offset_save_path)
if not os.path.exists(save_path_512_A):
    os.makedirs(save_path_512_A)
if not os.path.exists(save_path_512_A_off):
    os.makedirs(save_path_512_A_off)
if not os.path.exists(save_path_512_B):
    os.makedirs(save_path_512_B)
if not os.path.exists(save_path_512_label):
    os.makedirs(save_path_512_label)

txt_path = r'E:\JWDataSets\LEVIR-CD\LEVIR-CD-512-OFFSET\offsets_test.txt'
with open(txt_path, "w") as f:
    for idx in range(len(A_files)):
        A = cv2.imread(A_files[idx])
        name = A_files[idx].split('\\')[-1]
        B = cv2.imread(os.path.join(B_path,name))
        lab = cv2.imread(os.path.join(lab_path,name),-1)


        offsets = np.random.randint(-50, 50, (4, 2))  # 四个角点xy的偏移
        src_points = np.array([[0, 0], [0, 1024], [1024, 1024], [1024, 0]], np.float32)  # 定义原图四个角点
        offset_points = src_points + offsets  # 偏移后的四个角点
        src_H = cv2.getPerspectiveTransform(np.float32(offset_points), np.float32(src_points))  # 原图到偏移图的单应矩阵
        new_offset_points = np.dot(src_H,
                                   np.array([[0, 0, 1], [0, 1024, 1], [1024, 1024, 1], [1024, 0, 1]], np.float32).T).T
        # 对原始图像进行透视变换
        image_trans = cv2.warpPerspective(A, src_H, dsize=A.shape[:2])
        cv2.imwrite(os.path.join(offset_save_path,name+".png"), image_trans)
        n = 1
        for i in range(2):
            for j in range(2):
                # 计算子图像原始角点及偏移后角点
                sub_A = A[i*512:(i+1)*512,j*512:(j+1)*512,:]
                sub_B = B[i*512:(i+1)*512,j*512:(j+1)*512,:]
                sub_lab = lab[i*512:(i+1)*512,j*512:(j+1)*512]
                cv2.imwrite(os.path.join(save_path_512_A,name.split('.png')[0]+"_"+str(n)+".png"),sub_A)
                cv2.imwrite(os.path.join(save_path_512_B,name.split('.png')[0]+"_"+str(n)+".png"),sub_B)
                cv2.imwrite(os.path.join(save_path_512_label,name.split('.png')[0]+"_"+str(n)+".png"),sub_lab)
                sub_src_points = np.array([[j*512,i*512],[j*512,(i+1)*512],[(j+1)*512,i*512],[(j+1)*512,(i+1)*512]])
                sub_src_points_ = np.hstack([sub_src_points, np.ones((4, 1))])
                sub_offset_points = np.dot(src_H, sub_src_points_.T).T
                sub_offset_points = sub_offset_points[:, :2] / sub_offset_points[:, 2:]
                # 计算子图像偏移量,并写入txt
                sub_offsets = sub_offset_points - sub_src_points
                f.write(os.path.join(name.split('.png')[0]+"_"+str(n)+".png"))
                f.write(f"\n")
                for sub_offset in sub_offsets:
                    f.write(f"{sub_offset[0]},{sub_offset[1]}\n")
                # 裁剪子图像

                sub_img_trans = image_trans[i*512:(i+1)*512,j*512:(j+1)*512,:]
                cv2.imwrite(os.path.join(save_path_512_A_off,name.split('.png')[0]+"_"+str(n)+".png"),sub_img_trans)
                n+=1
        print(name)




# files = glob.glob(os.path.join(r"E:\JWDataSets\LEVIR-CD\LEVIR-CD-512-OFFSET\A", '*.png'))
# ftrain = open(r'E:\JWDataSets\LEVIR-CD\LEVIR-CD-512-OFFSET\list\train.txt', 'w')
# ftest = open(r'E:\JWDataSets\LEVIR-CD\LEVIR-CD-512-OFFSET\list\test.txt', 'w')
# fval = open(r'E:\JWDataSets\LEVIR-CD\LEVIR-CD-512-OFFSET\list\val.txt', 'w')
#     # 将文件名逐行写入文件
# for file in files:
#     if "train" in file:
#         name = file.split('\\')[-1]
#         ftrain.write(name + '\n')
#     if "val" in file:
#         name = file.split('\\')[-1]
#         fval.write(name + '\n')
#     if "test" in file:
#         name = file.split('\\')[-1]
#         ftest.write(name + '\n')