import torch.nn as nn
import torch
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

class Trainer():
    def __init__(self, model, device):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        self.scaler = torch.amp.GradScaler() if device == 'cuda' else None
        self.device = device
        self.epochs = 15
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train_one_epoch(self, model, loader, epoch):
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_count = 0

        pbar = tqdm(loader, desc=f"  Train {epoch:2d}/{self.epochs}", leave=False, unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

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
            total_accuracy += (preds == labels).sum().item()
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

    def train(self, model, train_loader, val_loader):
        best_val_acc = 0.0
        best_epoch = -1
        ckpt_path = "./checkpoints/best_final_cnn.pt"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
        print(f"{Colors.BOLD}  Training  |  {self.epochs} epochs  |  {self.device.upper()}{Colors.END}")
        print(f"{Colors.BOLD}{'─'*60}{Colors.END}\n")

        start_time = time.perf_counter()

        for epoch in tqdm(range(1, self.epochs + 1), desc="Epochs", unit="epoch"):
            epoch_start = time.perf_counter()

            train_loss, train_acc = self.train_one_epoch(model, train_loader, epoch)
            val_loss, val_acc = self.evaluate(model, val_loader, epoch)

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
