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
                     GTSRB_TRAINING_PATH, RESNET_CHECKPOINT_PATH, RESNET_CHECKPOINT_PATH_2, RESNET_CHECKPOINT_PATH_3,
                     RESNET_CHECKPOINT_PATH_4)
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
            if checkpoint_path == RESNET_CHECKPOINT_PATH:
                if "conv1." in k or "bn1." in k or "fc." in k:
                    continue
                
                new_state_dict["resnet." + k] = v
            else:
                new_state_dict[k] = v
        
        try:
            strict_load = True if checkpoint_path == RESNET_CHECKPOINT_PATH_2 else False
            model.load_state_dict(new_state_dict, strict=strict_load)
            print(f"Loaded model weights from {checkpoint_path} (strict={strict_load}).")
            if not strict_load:
                print("Note: Some keys might be missing or unexpected due to architecture modifications (strict=False).")
        except RuntimeError as e:
            print(f"Error loading model state_dict from {checkpoint_path}: {e}")
            print("This usually happens if module names in checkpoint do not exactly match model architecture.")
            print("Attempting to load with partial matches...")
            
            model_state_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
            model_state_dict.update(pretrained_dict)
            model.load_state_dict(model_state_dict)
            print("Loaded partial model weights (mismatched/missing keys were skipped).")
            raise RuntimeError(f"Failed to load checkpoint weights: {e}") # Bỏ comment nếu muốn dừng lại ngay khi có lỗi
            
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting with pretrained ImageNet weights for ResNet backbone (if applicable).")
        print("WARNING: This might lead to sub-optimal results if this is not the first stage and no checkpoint was loaded.")

def configure_optimizer(model, base_lr, weight_decay, unfreeze_resnet_layers=None, resnet_lr_multiplier=0.1):
    if unfreeze_resnet_layers is None:
        unfreeze_resnet_layers = []

    for param in model.resnet.parameters():
        param.requires_grad = False
    
    params_to_train = []

    for param in model.stn.parameters():
        param.requires_grad = True
    params_to_train.append({'params': model.stn.parameters(), 'lr': base_lr})
    print(f"Added STN parameters to optimizer with LR: {base_lr:.6f}")

    for param in model.conv1_modified.parameters():
        param.requires_grad = True
    params_to_train.append({'params': model.conv1_modified.parameters(), 'lr': base_lr})
    print(f"Added conv1_modified parameters to optimizer with LR: {base_lr:.6f}")
    
    for param in model.bn1_modified.parameters():
        param.requires_grad = True
    params_to_train.append({'params': model.bn1_modified.parameters(), 'lr': base_lr})
    print(f"Added bn1_modified parameters to optimizer with LR: {base_lr:.6f}")

    for param in model.fc.parameters():
        param.requires_grad = True
    params_to_train.append({'params': model.fc.parameters(), 'lr': base_lr})
    print(f"Added fc parameters to optimizer with LR: {base_lr:.6f}")
    
    for layer_name in unfreeze_resnet_layers:
        module = getattr(model.resnet, layer_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
            
            trainable_module_params = [p for p in module.parameters() if p.requires_grad]
            if trainable_module_params:
                params_to_train.append({'params': trainable_module_params, 'lr': base_lr * resnet_lr_multiplier})
                print(f"Unfrozen and added to optimizer: resnet.{layer_name} with LR: {base_lr * resnet_lr_multiplier:.6f}")
            else:
                print(f"Warning: Layer 'resnet.{layer_name}' has no trainable parameters to add to optimizer.")
        else:
            print(f"Warning: Layer 'resnet.{layer_name}' not found in model.resnet for unfreezing.")

    return optim.Adam(params_to_train, weight_decay=weight_decay)

def print_optimizer_config(optimizer):
    print("\n--- Optimizer configuration ---")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Param group {i}:")
        print(f"    Learning rate = {param_group['lr']:.6f}")
        print(f"    Number of parameters = {sum(p.numel() for p in param_group['params'])}")
    print("-----------------------------\n")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

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

    model = ResNetWithSTN(num_classes=GTSRB_NUM_CLASSES, stn_filters=(16, 32), stn_fc_units=128, input_size=IMAGE_SIZE)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(
        patience=EARLY_STOPPING_PARAMS['patience'],
        delta=EARLY_STOPPING_PARAMS['delta'],
        verbose=EARLY_STOPPING_PARAMS['verbose']
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # --- CẤU HÌNH CHO TỪNG GIAI ĐOẠN ---
    # BẠN SẼ UNCOMMENT KHỐI CẤU HÌNH CỦA GIAI ĐOẠN MÀ BẠN MUỐN CHẠY
    # VÀ ĐẢM BẢO CÁC KHỐI KHÁC ĐƯỢC COMMENT.

    # --- GIAI ĐOẠN 1 CẤU HÌNH: CHỈ HUẤN LUYỆN STN, conv1_modified, bn1_modified, FC ---
    # current_stage_name = 'Stage 1: STN + Modified ResNet Head (conv1, bn1, fc)'
    # unfreeze_layers_in_stage = []
    # current_lr_multiplier = 1.0
    # resnet_lr_multiplier_for_stage = 0.1
    # checkpoint_to_load_for_this_stage = RESNET_CHECKPOINT_PATH
    # checkpoint_to_save_after_stage = RESNET_CHECKPOINT_PATH_2

    # # --- GIAI ĐOẠN 2 CẤU HÌNH: THÊM LAYER4 + LAYER3 ---
    # # Đây là giai đoạn bạn đang gặp vấn đề. Chạy lại với code đã sửa.
    # current_stage_name = 'Stage 2: Add Layer4 + Layer3'
    # unfreeze_layers_in_stage = ['layer4', 'layer3']
    # current_lr_multiplier = 0.1
    # resnet_lr_multiplier_for_stage = 0.1
                                        
    # checkpoint_to_load_for_this_stage = RESNET_CHECKPOINT_PATH_2
    # checkpoint_to_save_after_stage = RESNET_CHECKPOINT_PATH_3

    # --- GIAI ĐOẠN 3 CẤU HÌNH: THÊM LAYER2 + LAYER1 (FULL FINE-TUNING) ---
    current_stage_name = 'Stage 3: Add Layer2 + Layer1 (Full Fine-tuning)'
    unfreeze_layers_in_stage = ['layer2', 'layer1'] # Mở băng layer2 và layer1
    current_lr_multiplier = 0.01 # LR giảm 100 lần cho các lớp đã học
    resnet_lr_multiplier_for_stage = 0.1 # LR multiplier cho ResNet layers (layer2, layer1 sẽ có LR = LR_global * 0.01 * 0.1 = LR_global * 0.001)
    checkpoint_to_load_for_this_stage = RESNET_CHECKPOINT_PATH_3 # Tải checkpoint từ giai đoạn 2
    checkpoint_to_save_after_stage = RESNET_CHECKPOINT_PATH_4 # Cập nhật checkpoint sau giai đoạn này


    print(f"\n--- Running: {current_stage_name} (Epochs: {EPOCHS}) ---")

    load_checkpoint(model, checkpoint_to_load_for_this_stage, DEVICE)

    optimizer = configure_optimizer(
        model, 
        LR * current_lr_multiplier,
        WEIGHT_DECAY, 
        unfreeze_resnet_layers=unfreeze_layers_in_stage,
        resnet_lr_multiplier=resnet_lr_multiplier_for_stage
    )
    print_optimizer_config(optimizer)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=(LR * current_lr_multiplier) * 0.001)

    trainable_params_this_stage = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters for {current_stage_name}: {trainable_params_this_stage}")

    best_val_accuracy = 0.0
    best_epoch = 0
    
    for epoch_in_stage in range(EPOCHS):
        print(f'\nEpoch {epoch_in_stage+1}/{EPOCHS} (Stage: {current_stage_name})')
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc * 100)
        val_accuracies.append(val_acc * 100)

        early_stopper(val_loss, model, checkpoint_to_save_after_stage, val_accuracy=val_acc*100)
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch_in_stage + 1

        if early_stopper.early_stop:
            print('Early stopping triggered.')
            checkpoint_epoch = epoch_in_stage + 1
            break
        scheduler.step()

    print(f"\n--- {current_stage_name} Finished ---")
    print(f"Best model state for this stage saved to {checkpoint_to_save_after_stage}.")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o', color='orange')
    plt.title(f'Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='x', color='green')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', marker='x', color='red')
    plt.title(f'Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    if not os.path.exists(RESNET_FIGURE_PATH):
        os.makedirs(RESNET_FIGURE_PATH)

    plt.savefig(os.path.join(RESNET_FIGURE_PATH, f'training_history_{current_stage_name.replace(" ", "_").replace(":", "")}.png'))
    plt.show()

if __name__ == '__main__':
    main()