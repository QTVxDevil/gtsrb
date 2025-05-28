import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models.resnet50 import build_resnet50
from src.cfg import BATCH, DEVICE, RESNET_CHECKPOINT_PATH, GTSRB_NUM_CLASSES, RESNET_FIGURE_PATH
from dataloader.transform import get_transform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class GTSRBTestDataset(Dataset):
    def __init__(self, csv_path, images_dir, mode='test'):
        self.data = pd.read_csv(csv_path, sep=';')
        self.images_dir = images_dir
        self.transform = get_transform(train=(mode == 'test'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['Filename'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, row['Filename']

def test_gtsrb():
    csv_path = r"datasets/GTSRB_Traffic_Sign/GTSRB_Final_Test_Images/GTSRB/Final_Test/GT-final_test.test.csv"
    images_dir = r"datasets/GTSRB_Traffic_Sign/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
    ground_truth_path = r"datasets/GTSRB_Traffic_Sign/GTSRB_Final_Test_GT/GT-final_test.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    test_dataset = GTSRBTestDataset(csv_path, images_dir, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_resnet50(num_classes=GTSRB_NUM_CLASSES)
    model = model.to(DEVICE)
    if os.path.exists(RESNET_CHECKPOINT_PATH):
        checkpoint = torch.load(RESNET_CHECKPOINT_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {RESNET_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise RuntimeError("Failed to load checkpoint weights")
    else:
        raise FileNotFoundError(f"No checkpoint found at {RESNET_CHECKPOINT_PATH}")
    model.eval()

    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_path, sep=';')
    ground_truth = ground_truth.set_index('Filename')['ClassId'].to_dict()

    predictions = []
    filenames = []
    incorrect_images = []
    incorrect_labels = []
    incorrect_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, leave=True)
        for images, file_names in progress_bar:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            filenames.extend(file_names)
            # Collect incorrect predictions
            for i in range(len(file_names)):
                if predicted[i].item() != ground_truth[file_names[i]]:
                    incorrect_images.append(images[i].cpu())
                    incorrect_labels.append(ground_truth[file_names[i]])
                    incorrect_predictions.append(predicted[i].item())
            progress_bar.set_description("Testing")

    correct_labels = [ground_truth[filename] for filename in filenames]
    accuracy = accuracy_score(correct_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Calculate precision, recall, and F1-score
    report = classification_report(correct_labels, predictions, target_names=[f"Class {i}" for i in range(GTSRB_NUM_CLASSES)], output_dict=True)
    print("\nClassification Report:")
    print(classification_report(correct_labels, predictions, target_names=[f"Class {i}" for i in range(GTSRB_NUM_CLASSES)]))

    # Calculate and display average precision, recall, and F1-score
    avg_precision = report['weighted avg']['precision']
    avg_recall = report['weighted avg']['recall']
    avg_f1 = report['weighted avg']['f1-score']
    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    cm = confusion_matrix(correct_labels, predictions, labels=list(range(GTSRB_NUM_CLASSES)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(GTSRB_NUM_CLASSES)), yticklabels=list(range(GTSRB_NUM_CLASSES)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    cm_png_path = os.path.join(RESNET_FIGURE_PATH, "test_confusion_matrix.png")
    os.makedirs(RESNET_FIGURE_PATH, exist_ok=True)
    plt.savefig(cm_png_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_png_path}")
    
    num_correct = sum(1 for p, g in zip(predictions, correct_labels) if p == g)
    num_incorrect = len(predictions) - num_correct
    print(f"Number of Correct Predictions: {num_correct}")
    print(f"Number of Incorrect Predictions: {num_incorrect}")

    predictions_path = os.path.join(RESNET_FIGURE_PATH, "test_predictions_with_accuracy.csv")
    os.makedirs(RESNET_FIGURE_PATH, exist_ok=True)
    predictions_df = pd.DataFrame({'Filename': filenames, 'Prediction': predictions, 'GroundTruth': correct_labels})
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions with ground truth saved to {predictions_path}")

    if incorrect_images:
        num_to_display = min(25, len(incorrect_images))  
        plt.figure(figsize=(15, 15))
        for i in range(num_to_display):
            plt.subplot(5, 5, i + 1)
            image = incorrect_images[i].permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.title(f"True: {incorrect_labels[i]}, Pred: {incorrect_predictions[i]}")
            plt.axis('off')
        plt.tight_layout()

        incorrect_png_path = os.path.join(RESNET_FIGURE_PATH, "incorrect_predictions.png")
        plt.savefig(incorrect_png_path)
        plt.close()
        print(f"Incorrect predictions plot saved to {incorrect_png_path}")
if __name__ == "__main__":
    test_gtsrb()
