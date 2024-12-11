from models.ChangeRD import *
import torch
import os
import time
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
# Define a custom dataset class for loading the images and labels
class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.a_dir = os.path.join(root_dir, 'A')
        self.b_dir = os.path.join(root_dir, 'B')
        self.label_dir = os.path.join(root_dir, 'label')
        self.a_images = sorted(os.listdir(self.a_dir))
        self.b_images = sorted(os.listdir(self.b_dir))
        self.labels = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.a_images)

    def __getitem__(self, idx):
        a_path = os.path.join(self.a_dir, self.a_images[idx])
        b_path = os.path.join(self.b_dir, self.b_images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        a_image = cv2.imread(a_path)
        b_image = cv2.imread(b_path)
        label = Image.open(label_path).convert('L')

        if self.transform:
            a_image = self.transform(a_image)
            b_image = self.transform(b_image)
            # label = self.transform(label)

        return a_image, b_image

def init():
    net = ChangeRD()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('checkpoints/off_CD_ChangeRD_LEVIR_b8_lr0.0001_adamw_train_test_400_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim111_256/best_ckpt.pt', map_location=device)
    saved_weights = checkpoint['model_G_state_dict']
    new_state_dict = {}
    for k, v in saved_weights.items():
        if k.startswith('module.'):
            name = k[7:]  # remove the "module." prefix
        else:
            name = k
        new_state_dict[name] = v

    # create a new model and load the new state dict
    net.load_state_dict(new_state_dict)
    net = net.to(device)
    net = net.eval()
    return net, device

def predict_batch(net, device, data_loader, use_fp16=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ChangeDetectionDataset(root_dir='/data/jingwei/change_detection_RD/datasets/LEVIR-CD-512-OFFSET', transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    start_time = time.time()
    with torch.no_grad():
        iters = 0
        for a_image, b_image in data_loader:
            a_image, b_image = a_image.to(device), b_image.to(device)
            if use_fp16:
                with torch.cuda.amp.autocast():
                    output = net(a_image, b_image)
            else:
                output = net(a_image, b_image)
            iters += 1
            print(f'Processed {iters} images')
    end_time = time.time()
    print(f'Prediction time: {end_time - start_time:.2f} seconds')

if __name__ == "__main__":
    net, device = init()
    predict_batch(net, device, None, use_fp16=False)  # Set use_fp16=False for full precision
