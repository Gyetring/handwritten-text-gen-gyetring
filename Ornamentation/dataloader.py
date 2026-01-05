import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as F


class SkeletonDataset(Dataset):
    def __init__(self, input_dir, label_dir, 
                 augment = False, max_translate=5, scale_range=(0.9, 1.1),):
        self.input_dir = input_dir
        self.label_dir = label_dir
        
        self.augment = augment
        self.max_translate = max_translate
        self.scale_range = scale_range

        # 只取在 input_dir 中存在的 png 文件
        self.filenames = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(".png")
        ])

        self.totensor = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),   # -> [1, 64, 64], float32, [0,1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        input_path = os.path.join(self.input_dir, filename)
        label_path = os.path.join(self.label_dir, filename)

        # 读取图片
        input_img = Image.open(input_path)
        label_img = Image.open(label_path)

        if self.augment:
            translate_x = random.uniform(-self.max_translate, self.max_translate)
            translate_y = random.uniform(-self.max_translate, self.max_translate)
            scale = random.uniform(*self.scale_range)

            input_img = F.affine(
                input_img,
                angle=0.0,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=0.0,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )

            label_img = F.affine(
                label_img,
                angle=0.0,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=0.0,
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        # transform
        input_tensor = self.totensor(input_img)
        label_tensor = self.totensor(label_img)

        return input_tensor, label_tensor


def get_dataloader(
    input_dir="skeletons",
    label_dir="labels",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    augment = False
):
    dataset = SkeletonDataset(input_dir, label_dir, augment=augment)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
