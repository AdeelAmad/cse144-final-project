import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models

class UnlabeledTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name
    
class Tester():
    def __init__(self, val_tf, batch_size, ckpt_path, model, device):
        test_dir = "./data/test"
        test_dataset = UnlabeledTestDataset(test_dir=test_dir, transform=val_tf)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        predictions = []

        print("Generating predictions for submission...")
        with torch.no_grad():
            for images, img_ids in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                for img_id, pred in zip(img_ids, preds):
                    predictions.append({"ID": img_id, "Label": pred.item()})

        submission_df = pd.DataFrame(predictions)
        submission_df.to_csv("submission.csv", index=False)
        print("Saved submission.csv successfully!")