import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Market1501Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = set()

        for label, person_dir in enumerate(os.listdir(root_dir)):
            self.classes.add(label)
            person_images = os.listdir(os.path.join(root_dir, person_dir))
            self.images.extend([os.path.join(root_dir, person_dir, img) for img in person_images])
            self.labels.extend([label] * len(person_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label