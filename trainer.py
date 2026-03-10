import torch.nn as nn
import torch
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import v2

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

class Trainer():
    def __init__(self, model, device, epochs):
        self.modelObj = model
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler() if device == 'cuda' else None
        self.device = device
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        self.mixup = v2.CutMix(num_classes=100, alpha=0.5)
        self.cutmix = v2.MixUp(num_classes=100, alpha=0.5)
        self.mixup_cutmix = v2.RandomChoice([self.cutmix, self.mixup])


        self.setup_optimizer(lr=1e-3)

    def setup_optimizer(self, lr, t_max=None):
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.modelObj.model.parameters()), lr=lr, weight_decay=0.03)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max or self.epochs, eta_min=1e-6)

    def train_one_epoch(self, model, loader, epoch):
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_count = 0

        pbar = tqdm(loader, desc=f"  Train {epoch:2d}/{self.epochs}", leave=False, unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # inputs, labels = self.cutmix(inputs, labels)
            # inputs, labels = self.mixup(inputs, labels)
            if torch.rand(1).item() < 0.5:
                inputs, labels = self.mixup_cutmix(inputs, labels)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device):
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            if labels.ndim == 2:
                _, _labels = torch.max(labels, 1) 
            else:
                _labels = labels
            
            total_accuracy += (preds == _labels).sum().item()
            total_count += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss/total_count:.4f}", acc=f"{total_accuracy/total_count:.4f}")

        return total_loss / total_count, total_accuracy / total_count

    @torch.no_grad()
    def evaluate(self, model, loader, epoch):
        model.eval()
        total_loss = 0
        total_accuracy = 0
        total_count = 0

        pbar = tqdm(loader, desc=f"  Val   {epoch:2d}/{self.epochs}", leave=False, unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = model(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_accuracy += (preds == labels).sum().item()
            total_count += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss/total_count:.4f}", acc=f"{total_accuracy/total_count:.4f}")

        return total_loss / total_count, total_accuracy / total_count

    def train(self, train_loader, val_loader):
        best_val_acc = 0.0
        best_epoch = -1
        ckpt_path = "./checkpoints/best_final_cnn.pt"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        model = self.modelObj.model

        print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
        print(f"{Colors.BOLD}  Training  |  {self.epochs} epochs  |  {self.device.upper()}{Colors.END}")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

        start_time = time.perf_counter()

        for epoch in tqdm(range(1, self.epochs + 1), desc="Epochs", unit="epoch"):

            if epoch == 11:
                tqdm.write(f"\n{Colors.YELLOW}{Colors.BOLD}Stage 2 training...{Colors.END}\n")
                self.modelObj.stage_2_training()
                self.setup_optimizer(lr=1e-4, t_max=self.epochs - 10)

            if epoch == 21:
                tqdm.write(f"\n{Colors.YELLOW}{Colors.BOLD}Stage 3 training...{Colors.END}\n")
                self.modelObj.stage_3_training()
                self.setup_optimizer(lr=1e-5, t_max=self.epochs - 20)

            epoch_start = time.perf_counter()

            train_loss, train_acc = self.train_one_epoch(model, train_loader, epoch)
            val_loss, val_acc = self.evaluate(model, val_loader, epoch)

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            is_best = val_acc > best_val_acc
            marker = f"{Colors.GREEN}{Colors.BOLD}*{Colors.END}" if is_best else " "

            tqdm.write(
                f"{marker} [{epoch:2d}/{self.epochs}]  "
                f"train  loss={Colors.BLUE}{train_loss:.4f}{Colors.END}  acc={Colors.BLUE}{train_acc:.4f}{Colors.END}  |  "
                f"val  loss={Colors.YELLOW}{val_loss:.4f}{Colors.END}  acc={Colors.YELLOW}{val_acc:.4f}{Colors.END}  |  "
                f"{time.perf_counter() - epoch_start:.1f}s"
            )

            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch}, ckpt_path)

        total_time = time.perf_counter() - start_time
        print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}  Best val acc : {best_val_acc:.4f}  (epoch {best_epoch}){Colors.END}")
        print(f"{Colors.BOLD}  Checkpoint   : {ckpt_path}{Colors.END}")
        print(f"{Colors.BOLD}  Total time   : {total_time:.1f}s{Colors.END}")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

        return ckpt_path

    def curves(self):
        epochs_range = range(1, self.epochs + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history["train_loss"], label='Train Loss', marker='o')
        plt.plot(epochs_range, self.history["val_loss"], label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.history["train_acc"], label='Train Accuracy', marker='o')
        plt.plot(epochs_range, self.history["val_acc"], label='Validation Accuracy', marker='o')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
