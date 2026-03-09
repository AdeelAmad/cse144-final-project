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
        self._model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

        # Get the last output layer
        num_feat = self._model.classifier[2].in_features
        self._model.classifier = nn.Sequential(
            self._model.classifier[0],
            self._model.classifier[1],
            nn.Dropout(p=0.6),
            nn.Linear(num_feat, 100)
        )
        
        # Move to device
        self._model = self._model.to(device)

        self.stage_1_training()

        # Parameter count (Total vs Trainable)
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        print(f"{Colors.BOLD}{Colors.GREEN}Total params:{Colors.END} {total_params:,}")
        print(f"{Colors.BOLD}{Colors.GREEN}Trainable params:{Colors.END} {trainable_params:,}")

    def stage_1_training(self):
        for param in self._model.parameters():
                param.requires_grad = False

        for param in self._model.classifier.parameters():
            param.requires_grad = True

    def stage_2_training(self):
         for param in self._model.features[7].parameters():
            param.requires_grad = True
         for block in self._model.features[5][-3]:
            for param in block.parameters():
                param.requires_grad = True

    @property
    def model(self):
        return self._model