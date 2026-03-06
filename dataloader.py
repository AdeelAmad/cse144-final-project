import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import kagglehub

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'
    OFF = '\033[0m'


'''
    Handles downloading, transforming, and loading the data
    Returns two dataloaders
    train_loader
    val_loader
'''
class DatasetLoader():
    def __init__(self):
        print(f"{Colors.BLUE}Fetching competition data...{Colors.END}")
        kagglehub.login()
        try:
            path = kagglehub.competition_download('ucsc-cse-144-winter-2026-final-project')
            print(f"{Colors.GREEN}Data ready at: {Colors.BOLD}{path}{Colors.END}")
        except Exception as e:
            print(f"\033[91mError during download: {e}\033[0m")
            raise

        # Important variables + seed
        data_dir = os.path.join(path, 'train')
        batch_size = 512
        num_workers = 0
        train_val_split = 0.8
        torch.manual_seed(42)

        # Training dataset transforms
        # Performing random crop, flips, color jitter, rotation
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Validation tranforms
        val_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Fixes lexicographical class names issue
        class NumericalImageFolder(datasets.ImageFolder):
            def find_classes(self, directory):
                classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))], key=int)
                class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
                return classes, class_to_idx

        train_dataset_full = NumericalImageFolder(root=data_dir, transform=train_tf)
        val_dataset_full = NumericalImageFolder(root=data_dir, transform=val_tf)

        # Create train val split
        total_data = len(train_dataset_full)
        indices = torch.randperm(total_data).tolist()
        split = int(train_val_split * total_data)

        train_indices = indices[:split]
        val_indices = indices[split:]

        # Subset the dataset
        train_set = Subset(train_dataset_full, train_indices)
        val_set = Subset(val_dataset_full, val_indices)

        # Dataloaders
        self._train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self._val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"{Colors.BOLD}{Colors.BLUE}Total images:{Colors.END} {total_data}")
        print(f"{Colors.BOLD}{Colors.BLUE}Training set:{Colors.END} {len(train_set)}")
        print(f"{Colors.BOLD}{Colors.BLUE}Validation set:{Colors.END} {len(val_set)}")

    @property
    def train_loader(self):
        return self._train_loader
    
    @property
    def val_loader(self):
        return self._val_loader