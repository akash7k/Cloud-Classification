
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from config import batchsize, resize_x, resize_y

# Dataset class to apply transform
class ApplyTransform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.transform(x)
        return x, y

# Dataloader function
def unicornLoader(data_path):
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Default ImageNet stats; you can modify if you calculated mean/std
    ])

    test_transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_path)
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))), test_size=0.1, stratify=[y for _, y in full_dataset]
    )

    train_dataset = ApplyTransform(Subset(full_dataset, train_indices), transform=train_transform)
    val_dataset = ApplyTransform(Subset(full_dataset, val_indices), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
