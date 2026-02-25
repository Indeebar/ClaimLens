import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import CarDamageDataset, TRAIN_TRANSFORMS, VAL_TRANSFORMS
from model import DamageClassifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw/damage_images')
SAVE_PATH = 'best_model.pt'

def train():
    full_ds = CarDamageDataset(DATA_DIR, transform=TRAIN_TRANSFORMS)
    n_val = int(len(full_ds) * 0.2)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    val_ds.dataset.transform = VAL_TRANSFORMS

    # Fix: num_workers=0 for Windows to avoid multiprocessing issues with DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DamageClassifier(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc, patience_counter = 0.0, 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        scheduler.step()
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save(SAVE_PATH)
            print(f'  Saved best model (val_acc={val_acc:.3f})')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print('Early stopping triggered.')
                break

if __name__ == '__main__':
    train()
