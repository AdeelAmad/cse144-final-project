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

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'
    OFF = '\033[0m'


def main():
    print(f"{Colors.BLUE}Torch:{Colors.END}", torch.__version__)
    device = "cpu"
    print(f"{Colors.BLUE}Device:{Colors.END}", device)
    


if __name__ == "__main__":
    main()