import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
    
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

class Tester():
    def __init__(self, batch_size, ckpt_path, model, device):
        val_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dir = "./data/test"
        test_dataset = UnlabeledTestDataset(test_dir=test_dir, transform=val_tf)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
        print(f"{Colors.BOLD}  Inference  |  {len(test_dataset)} images  |  {device.upper()}{Colors.END}")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}")

        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        predictions = []

        with torch.no_grad():
            for images, img_ids in tqdm(test_loader, desc="  Predicting", unit="batch"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                for img_id, pred in zip(img_ids, preds):
                    predictions.append({"ID": img_id, "Label": pred.item()})

        submission_df = pd.DataFrame(predictions)
        submission_df.to_csv("submission.csv", index=False)
        print(f"{Colors.GREEN}{Colors.BOLD}  Saved submission.csv  ({len(predictions)} predictions){Colors.END}")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")