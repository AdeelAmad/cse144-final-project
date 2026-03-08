import torch.nn as nn
from torchvision import models
import torch

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'
    OFF = '\033[0m'

class Model():
    def __init__(self, device):
        # Load model with weights
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

        # Freeze all parameteres
        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[7].parameters():
            param.requires_grad = True

        # Get the last output layer
        num_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_feat, 100)
        )

        # Replace the last layer with a new one with 100 classes
        
        # Move to device
        self._model = model.to(device)

        # Parameter count (Total vs Trainable)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{Colors.BOLD}{Colors.GREEN}Total params:{Colors.END} {total_params:,}")
        print(f"{Colors.BOLD}{Colors.GREEN}Trainable params:{Colors.END} {trainable_params:,}")

    @property
    def model(self):
        return self._model