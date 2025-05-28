import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from dataloader.gtsrb_dataloader import GTSRB_load
from models.resnet_stn import ResNetWithSTN
from src.cfg import (BATCH, LR, GTSRB_NUM_CLASSES, DEVICE, WEIGHT_DECAY,
                     EPOCHS, EARLY_STOPPING_PARAMS, RESNET_FIGURE_PATH, IMAGE_SIZE,
                     GTSRB_TRAINING_PATH, RESNET_CHECKPOINT_PATH, RESNET_CHECKPOINT_PATH_2)
from src.earlystoping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR

def load_checkpoint(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if "conv1." in k and "conv1_modified" not in k:
                continue
            if "bn1." in k and "bn1_modified" not in k:
                continue
            if "stn." in k:
                continue
            if "fc." in k:
                continue
            
            if k.startswith("resnet."):
                k = k[len("resnet."):]
            
            new_state_dict[k] = v
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded model weights from {checkpoint_path} (mismatched/skipped layers ignored).")
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting with pretrained ImageNet weights for ResNet backbone.")
        

def configure_optimizer(model, lr, weight_decay):
    return optim.Adam([
        {'params': model.conv1_modified.parameters(), 'lr': lr * 0.1},
        {'params': model.bn1_modified.parameters(), 'lr': lr * 0.1},
        {'params': model.stn.parameters(), 'lr': lr * 0.1},
        {'params': model.fc.parameters(), 'lr': lr * 0.1}
    ], weight_decay=WEIGHT_DECAY)

def print_optimizer_config(optimizer):
    print("Starting fine-tuning with the following optimizer configuration:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Param group {i}: Learning rate = {param_group['lr']}, Weight decay = {param_group.get('weight_decay', 0)}")
        
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

    model = ResNetWithSTN(num_classes=GTSRB_NUM_CLASSES, input_size=IMAGE_SIZE)
    model = model.to(DEVICE)

    load_checkpoint(model, RESNET_CHECKPOINT_PATH, DEVICE)

    model.unfreeze_layers(layer_names=['conv1_modified', 'bn1_modified', 'stn', 'fc'])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    optimizer = configure_optimizer(model, LR, WEIGHT_DECAY)
    print_optimizer_config(optimizer)
    
    optimizer = configure_optimizer(model, LR, WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.001)

    early_stopper = EarlyStopping(
        patience=EARLY_STOPPING_PARAMS['patience'],
        delta=EARLY_STOPPING_PARAMS['delta'],
        verbose=EARLY_STOPPING_PARAMS['verbose']
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    checkpoint_epoch = None

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        early_stopper(val_loss, model, RESNET_CHECKPOINT_PATH_2, val_accuracy=val_acc*100)
        
        if early_stopper.early_stop:
            print('Early stopping triggered.')
            checkpoint_epoch = epoch + 1
            break
        
        scheduler.step()

    if checkpoint_epoch is not None:
        train_losses = train_losses[:checkpoint_epoch]
        val_losses = val_losses[:checkpoint_epoch]
        train_accuracies = train_accuracies[:checkpoint_epoch]
        val_accuracies = val_accuracies[:checkpoint_epoch]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', color='blue')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='x', color='green')
    ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', marker='x', color='red')
    ax2.set_ylabel('Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title('Training and Validation Loss and Accuracy')
    plt.tight_layout()
    
    # Tạo thư mục figure nếu chưa có
    if not os.path.exists(RESNET_FIGURE_PATH):
        os.makedirs(RESNET_FIGURE_PATH)

    plt.savefig(os.path.join(RESNET_FIGURE_PATH, 'training_history.png'))
    plt.show()


if __name__ == '__main__':
    main()