import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models.resnet_stn import ResNetWithSTN
from src.cfg import BATCH, DEVICE, RESNET_CHECKPOINT_PATH_4, GTSRB_NUM_CLASSES, RESNET_FIGURE_PATH, IMAGE_SIZE
from dataloader.transform import get_transform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict, Counter

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
    csv_path = r"D:\USTH_SUBJECTS\B3\project\gtsrb\datasets\GTSRB_Traffic_Sign\GTSRB_Final_Test_Images\GTSRB\Final_Test\GT-final_test.test.csv"
    images_dir = r"D:\USTH_SUBJECTS\B3\project\gtsrb\datasets\GTSRB_Traffic_Sign\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images"
    ground_truth_path = r"D:\USTH_SUBJECTS\B3\project\gtsrb\datasets\GTSRB_Traffic_Sign\GTSRB_Final_Test_GT\GT-final_test.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    test_dataset = GTSRBTestDataset(csv_path, images_dir, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    model = ResNetWithSTN(num_classes=GTSRB_NUM_CLASSES, input_size=IMAGE_SIZE)
    model = model.to(DEVICE)
    if os.path.exists(RESNET_CHECKPOINT_PATH_4):
        checkpoint = torch.load(RESNET_CHECKPOINT_PATH_4, map_location=DEVICE)
        try:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {RESNET_CHECKPOINT_PATH_4}")
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise RuntimeError("Failed to load checkpoint weights")
    else:
        raise FileNotFoundError(f"No checkpoint found at {RESNET_CHECKPOINT_PATH_4}")
    model.eval()

    ground_truth_df = pd.read_csv(ground_truth_path, sep=';')
    ground_truth_map = ground_truth_df.set_index('Filename')['ClassId'].to_dict()

    all_preds = []
    all_labels = []
    all_filenames = []
    incorrect_images = []
    incorrect_labels = []
    incorrect_predictions = []
    misclassified_samples = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, leave=True)
        for images, file_names in progress_bar:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted_batch = torch.max(outputs, 1)

            for i in range(len(file_names)):
                filename = file_names[i]
                true_label = ground_truth_map[filename]
                predicted_label = predicted_batch[i].item()

                all_preds.append(predicted_label)
                all_labels.append(true_label)
                all_filenames.append(filename)

                if predicted_label != true_label:
                    incorrect_images.append(images[i].cpu())
                    incorrect_labels.append(true_label)
                    incorrect_predictions.append(predicted_label)
                    misclassified_samples.append({
                        'video_path': filename, # Renamed to filename for GTSRB context
                        'true_label': true_label,
                        'pred_label': predicted_label
                    })
            progress_bar.set_description("Testing")

    # --- Standard Metrics Calculation ---
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Create class_names list for GTSRB
    # Assuming GTSRB_NUM_CLASSES is defined in src.cfg
    gtsrb_class_names = [f"Class {i}" for i in range(GTSRB_NUM_CLASSES)]

    report = classification_report(all_labels, all_preds, target_names=gtsrb_class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=gtsrb_class_names))

    avg_precision = report['weighted avg']['precision']
    avg_recall = report['weighted avg']['recall']
    avg_f1 = report['weighted avg']['f1-score']
    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(GTSRB_NUM_CLASSES)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gtsrb_class_names, yticklabels=gtsrb_class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Overall Confusion Matrix')
    cm_png_path = os.path.join(RESNET_FIGURE_PATH, "test_confusion_matrix_phase4.png")
    os.makedirs(RESNET_FIGURE_PATH, exist_ok=True)
    plt.savefig(cm_png_path)
    plt.close()
    print(f"Overall Confusion matrix saved to {cm_png_path}")
    
    num_correct = sum(1 for p, g in zip(all_preds, all_labels) if p == g)
    num_incorrect = len(all_preds) - num_correct
    print(f"Number of Correct Predictions: {num_correct}")
    print(f"Number of Incorrect Predictions: {num_incorrect}")

    predictions_path = os.path.join(RESNET_FIGURE_PATH, "test_predictions_with_accuracy_phase4.csv")
    os.makedirs(RESNET_FIGURE_PATH, exist_ok=True)
    predictions_df = pd.DataFrame({'Filename': all_filenames, 'Prediction': all_preds, 'GroundTruth': all_labels})
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions with ground truth saved to {predictions_path}")

    if incorrect_images:
        num_to_display = min(25, len(incorrect_images))  
        plt.figure(figsize=(15, 15))
        for i in range(num_to_display):
            plt.subplot(5, 5, i + 1)
            image = incorrect_images[i].permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.title(f"True: {gtsrb_class_names[incorrect_labels[i]]}, Pred: {gtsrb_class_names[incorrect_predictions[i]]}")
            plt.axis('off')
        plt.tight_layout()

        incorrect_png_path = os.path.join(RESNET_FIGURE_PATH, "incorrect_predictions_phase4.png")
        plt.savefig(incorrect_png_path)
        plt.close()
        print(f"Incorrect predictions plot saved to {incorrect_png_path}")

    # --- PPlotting Confusion Matrix for Top 5 Incorrectly Classified Classes ---

    incorrect_counts = defaultdict(int)
    for true, pred in zip(all_labels, all_preds):
        if true != pred:
            incorrect_counts[true] += 1

    sorted_incorrect_counts = sorted(incorrect_counts.items(), key=lambda item: item[1], reverse=True)
    top_5_incorrect_class_indices = [class_id for class_id, count in sorted_incorrect_counts[:5]]
    top_5_incorrect_class_names = [gtsrb_class_names[idx] for idx in top_5_incorrect_class_indices]

    print(f"\nTop 5 classes by number of Incorrect Classifications:")
    for cls_idx in top_5_incorrect_class_indices:
        print(f"  {gtsrb_class_names[cls_idx]}: {incorrect_counts[cls_idx]} incorrect classifications")

    true_filtered_top5_incorrect = [
        true for true, pred in zip(all_labels, all_preds)
        if true in top_5_incorrect_class_indices
    ]
    pred_filtered_top5_incorrect = [
        pred for true, pred in zip(all_labels, all_preds)
        if true in top_5_incorrect_class_indices
    ]

    if not true_filtered_top5_incorrect:
        print("No valid predictions for top 5 incorrectly classified classes.")
    else:
        all_misclassified_preds_from_top5_true = []
        for sample in misclassified_samples:
            if sample['true_label'] in top_5_incorrect_class_indices:
                all_misclassified_preds_from_top5_true.append(sample['pred_label'])

        pred_counter_top5_incorrect = Counter(all_misclassified_preds_from_top5_true)
        
        dynamic_pred_classes = [
            cls for cls, _ in pred_counter_top5_incorrect.most_common()
            if cls not in top_5_incorrect_class_indices
        ][:7]

        cm_pred_indices = top_5_incorrect_class_indices + dynamic_pred_classes
        cm_label_map = {cls: i for i, cls in enumerate(top_5_incorrect_class_indices)}
        cm_pred_map = {cls: i for i, cls in enumerate(cm_pred_indices)}

        cm_matrix_top5_incorrect = [[0 for _ in range(len(cm_pred_indices))] for _ in range(len(top_5_incorrect_class_indices))]

        for true, pred in zip(true_filtered_top5_incorrect, pred_filtered_top5_incorrect):
            if true in cm_label_map and pred in cm_pred_map:
                row = cm_label_map[true]
                col = cm_pred_map[pred]
                cm_matrix_top5_incorrect[row][col] += 1

        row_labels_top5_incorrect = [gtsrb_class_names[idx] for idx in top_5_incorrect_class_indices]
        col_labels_top5_incorrect = [gtsrb_class_names[idx] for idx in cm_pred_indices]

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm_matrix_top5_incorrect, annot=True, fmt='d', cmap='Reds',
                    xticklabels=col_labels_top5_incorrect, yticklabels=row_labels_top5_incorrect)
        plt.title("Confusion Matrix - Top 5 Incorrectly Classified Classes")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_top5_incorrect_png_path = os.path.join(RESNET_FIGURE_PATH, "test_confusion_matrix_top5_incorrect_custom.png")
        plt.savefig(cm_top5_incorrect_png_path)
        plt.close()
        print(f"Confusion matrix for top 5 incorrectly classified classes saved to {cm_top5_incorrect_png_path}")

        print("\nMisclassification details for Top 5 Incorrectly Classified Classes:")
        for orig_cls_idx in top_5_incorrect_class_indices:
            true_name = gtsrb_class_names[orig_cls_idx]

            misclassified_preds_for_this_true_class = [
                sample['pred_label'] for sample in misclassified_samples
                if sample['true_label'] == orig_cls_idx and sample['pred_label'] != orig_cls_idx
            ]
            if not misclassified_preds_for_this_true_class:
                print(f"  {true_name}: No misclassifications")
                continue

            counts = Counter(misclassified_preds_for_this_true_class)
            print(f"  {true_name} misclassified as:")
            for mis_cls_idx, count in counts.most_common():
                pred_name = gtsrb_class_names[mis_cls_idx]
                label_info = "(among top 5 incorrect true classes)" if mis_cls_idx in top_5_incorrect_class_indices else ""
                print(f"    {pred_name}: {count} times {label_info}")

        print("\nMisclassified video files for Top 5 Incorrectly Classified Classes:")
        for cls_idx in top_5_incorrect_class_indices:
            cls_name = gtsrb_class_names[cls_idx]
            print(f"\nFiles for true class '{cls_name}':")
            for sample in misclassified_samples:
                if sample['true_label'] == cls_idx:
                    pred_name = gtsrb_class_names[sample['pred_label']]
                    print(f"  Predicted as '{pred_name}' --> {sample['video_path']}")

if __name__ == "__main__":
    test_gtsrb()