"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils import data
import cv2
from datasets.data_utils import CDDataAugmentation
import torch

"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  #train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)
        
        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform
        self.is_train = is_train

    def __getitem__(self, index):
        name = self.img_name_list[index]
        # print(name)
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        label = np.array(Image.open(L_path), dtype=np.uint8)
        # if you are getting error because of dim mismatch ad [:,:,0] at the end

        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255
        [img_A, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        # 生成扭曲标签及扭曲图像
        off_label = self.gen_off_label()
        src_points = np.array([[0, 0], [0, 512], [512, 0], [512, 512]], np.float32)  # 定义原图四个角点
        offset_points = src_points + off_label
        img_A = np.array(img_A)
        if self.is_train:
            H = cv2.getPerspectiveTransform(np.float32(offset_points), np.float32(src_points))
            img_O = cv2.warpPerspective(img_A, H, dsize=img_A.shape[:2])
        else:
            img_O = img_A.copy()
        img_O = TF.to_tensor(img_O)
        img_O = TF.normalize(img_O, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_B = TF.to_tensor(img_B)
        img_B = TF.normalize(img_B, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_A = TF.to_tensor(img_A)
        img_A = TF.normalize(img_A, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                
        label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)

        return {'name': name, 'A': img_O, 'B': img_B, 'L': label, 'OFF_L': torch.tensor(off_label.reshape(-1)), 'S_A': img_A}

    def gen_off_label(self):
        random_bias_np = np.random.randint(-50, 50, (4, 2))
        return np.array(random_bias_np,np.float32)