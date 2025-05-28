import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from dataloader.gtsrb_dataloader import GTSRB_load
from models.resnet50 import build_resnet50
from src.cfg import (GTSRB_NUM_CLASSES, DEVICE, GTSRB_TRAINING_PATH, IMAGE_SIZE, BATCH, RESNET_CHECKPOINT_PATH)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Val', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
    accuracy = correct / total
    return accuracy, np.array(all_preds)

def main():
    # Load test set
    test_dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    # Load model
    model = build_resnet50(num_classes=GTSRB_NUM_CLASSES)
    model.load_state_dict(torch.load(RESNET_CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    # Evaluate
    accuracy, preds = evaluate(model, test_loader, DEVICE)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

if __name__ == '__main__':
    main()
