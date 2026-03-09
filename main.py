import torch
from dataloader import DatasetLoader
from model import Model
from trainer import Trainer
from tester import Tester

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'
    OFF = '\033[0m'


def main():
    device = "mps"
    batch_size = 32

    print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
    print(f"{Colors.BOLD}  CSE144 Final Project{Colors.END}")
    print(f"{Colors.BOLD}{'─'*60}{Colors.END}")
    print(f"  {Colors.BLUE}Torch  :{Colors.END} {torch.__version__}")
    print(f"  {Colors.BLUE}Device :{Colors.END} {device.upper()}")
    print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

    datasetloader = DatasetLoader(batch_size)

    train_data = datasetloader.train_loader.dataset
    
    if hasattr(train_data, 'dataset'):
        train_data = train_data.dataset 
        
    idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}

    modelObj = Model(device)

    trainer = Trainer(modelObj, device)
    ckpt_path = trainer.train(datasetloader.train_loader, datasetloader.val_loader)
    trainer.curves()

    Tester(batch_size, ckpt_path, modelObj, device, idx_to_class)



if __name__ == "__main__":
    main()