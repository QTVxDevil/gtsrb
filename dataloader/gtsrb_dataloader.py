import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from dataloader.transform import get_transform

class GTSRB_load(Dataset):
    def __init__(self, training_dir, mode='train'):
        self.training_dir = training_dir
        if mode not in ['train', 'val']:
            raise ValueError("mode must be 'train' or 'val'")
        self.transform = get_transform(train=(mode == 'train'))
        self.data = []
        self.labels = []

        for class_id in os.listdir(training_dir):
            class_dir = os.path.join(training_dir, class_id)
            csv_file = os.path.join(class_dir, f'GT-{class_id}.csv')
            if os.path.isdir(class_dir) and os.path.isfile(csv_file):
                try:
                    df = pd.read_csv(csv_file, sep=';')
                    for _, row in df.iterrows():
                        img_path = os.path.join(class_dir, row['Filename'])
                        if os.path.isfile(img_path):
                            self.data.append(img_path)
                            self.labels.append(int(row['ClassId']))
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue

        if not self.data:
            raise ValueError("No valid data found in the dataset directory")

        train_data, val_data, train_labels, val_labels = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42
        )

        if mode == 'train':
            self.data, self.labels = train_data, train_labels
        else:
            self.data, self.labels = val_data, val_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_path = self.data[idx]
        except IndexError:
            raise IndexError(f"Index {idx} is out of range for dataset with length {len(self.data)}")

        try:
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise