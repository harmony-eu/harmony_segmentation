import os
import random
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode as im

class CustomDataset(Dataset):
    def __init__(self, path_to_imgs, path_to_sem_masks, transform=True):
        self.path_to_imgs = path_to_imgs
        self.path_to_masks = path_to_sem_masks
        self.images = os.listdir(self.path_to_imgs)
        self.masks = os.listdir(self.path_to_masks)
        self.transform = transform
        self.images.sort()
        self.masks.sort()
        assert len(self.images) == len(self.masks)

        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.blur = torchvision.transforms.GaussianBlur(3)
        self.jitter = torchvision.transforms.ColorJitter()
        self.resize_image = torchvision.transforms.Resize((480, 640), interpolation=im.BILINEAR)
        self.resize_mask = torchvision.transforms.Resize((480, 640), interpolation=im.NEAREST)

    def __getitem__(self, item):
        img = os.path.join(self.path_to_imgs, self.images[item])
        mask = os.path.join(self.path_to_masks, self.masks[item])
        image = torchvision.io.read_image(img)
        sem_mask = torchvision.io.read_image(mask)

        # resize
        image = self.resize_image(image)
        sem_mask = self.resize_mask(sem_mask)

        # data augmentation
        if self.transform:
            # horizontal flip
            p_hflip = torch.rand(1)
            if p_hflip > 0.5:
                image = self.hflip(image)
                sem_mask = self.hflip(sem_mask)
            p_vflip = torch.rand(1)
            if p_vflip > 0.5:
                image = self.vflip(image)
                sem_mask = self.vflip(sem_mask)
            # gaussian blur
            # p_blur = torch.rand(1)
            # if p_blur > 0.5:
            #     image = self.blur(image)
            # color jittering
            p_jitter = torch.rand(1)
            if p_jitter > 0.5:
                image = self.jitter(image)

        sem_mask -= 1
        sem_mask[sem_mask == -1] = 0
        # sem_mask[sem_mask == 2] = 0
        sem_mask[sem_mask == 3] = 0
        sem_mask[sem_mask == 4] = 0
        sem_mask[sem_mask == 5] = 3
        sem_mask[sem_mask == 6] = 0

        return image, sem_mask

    def __len__(self):
        return len(self.images)