import torch.nn as nn
import torch
import os
import time
import matplotlib as plt

class Trainer():
    def __init__(self, model, device):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        self.scaler = torch.amp.GradScaler(device)
        self.device = device

    def train_one_epoch(self, model, loader):
        model.train()
        total_loss = 0

        total_accuracy = 0
        total_count = 0

        for inputs, labels in loader:

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            total_accuracy += (preds == labels).sum().item()
            total_count += labels.size(0)

        avg_loss = total_loss / total_count
        accuracy = total_accuracy / total_count

        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, model, loader):
        """Evaluate model and return (avg_loss, accuracy)."""
        model.eval()

        total_loss = 0
        total_accuracy = 0
        total_count = 0

        for inputs, labels in loader:

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = model(inputs)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            total_accuracy += (preds == labels).sum().item()
            total_count += labels.size(0)

        avg_loss = total_loss / total_count
        accuracy = total_accuracy / total_count

        return avg_loss, accuracy
    
    def train(self, model, train_loader, val_loader):
        self.epochs = 15
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        best_epoch = -1
        ckpt_path = "./checkpoints/best_final_cnn.pt"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        start_time = time.perf_counter()

        for epoch in range(self.epochs):
            epoch_start = time.perf_counter()

            train_loss, train_acc = self.train_one_epoch(model, train_loader)
            val_loss, val_acc = self.evaluate(model, val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}] | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Time for epoch: {time.perf_counter() - epoch_start:.2f} sec, Time Elapsed: {time.perf_counter() - start_time:.2f} sec")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch
                }, ckpt_path)

        # ========== YOUR CODE ENDS HERE ============

        print("Best val acc:", best_val_acc, "at epoch", best_epoch)
        print("Saved to:", ckpt_path)
    
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

        # ========== YOUR CODE ENDS HERE ============