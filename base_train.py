import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import random
import numpy as np
from dataloader.gtsrb_dataloader import GTSRB_load
from models.resnet50 import build_resnet50
from src.cfg import (BATCH, LR, GTSRB_NUM_CLASSES, DEVICE,
                     EPOCHS, EARLY_STOPPING_PARAMS,
                     GTSRB_TRAINING_PATH, RESNET_CHECKPOINT_PATH)
from src.earlystoping import EarlyStopping

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, leave=True)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        current_accuracy = 100.0 * correct / total if total > 0 else 0.0
        current_loss = running_loss / total if total > 0 else 0.0
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_accuracy:.2f}%")
        
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Val', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    
    train_dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='train')
    val_dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_resnet50(num_classes=GTSRB_NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    early_stopper = EarlyStopping(
        patience=EARLY_STOPPING_PARAMS['patience'],
        delta=EARLY_STOPPING_PARAMS['delta'],
        verbose=EARLY_STOPPING_PARAMS['verbose']
    )
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
        early_stopper(val_loss, model, RESNET_CHECKPOINT_PATH, val_accuracy=val_acc*100)
        if early_stopper.early_stop:
            print('Early stopping triggered.')
            break

if __name__ == '__main__':
    main()
