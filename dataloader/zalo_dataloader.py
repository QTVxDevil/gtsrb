import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from dataloader.transform import get_transform

class ZaloProcessedDataset(Dataset):
    def __init__(self, dataset_dir, mode='train'):
        self.dataset_dir = dataset_dir
        if mode not in ['train', 'val', 'test']:
            raise ValueError("mode must be 'train', 'val', or 'test'")
        self.transform = get_transform(train=(mode == 'train'))
        self.data = []
        self.labels = []

        for label in os.listdir(dataset_dir):
            label_dir = os.path.join(dataset_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.data.append(img_path)
                    self.labels.append(int(label))

        if not self.data:
            raise ValueError("No valid data found in the dataset directory")
        
        # Split into train, val, test (7:2:1)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            self.data, self.labels, test_size=0.3, random_state=42, stratify=self.labels
        )
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=1/3, random_state=42, stratify=temp_labels
        )

        if mode == 'train':
            self.data, self.labels = train_data, train_labels
        elif mode == 'val':
            self.data, self.labels = val_data, val_labels
        elif mode == 'test':
            self.data, self.labels = test_data, test_labels
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = None
        try:
            img_path = self.data[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label - 1
        except IndexError:
            raise IndexError(f"Index {idx} is out of range for dataset with length {len(self.data)}")
        except Exception as e:
            if img_path is not None:
                print(f"Error loading image {img_path}: {e}")
            else:
                print(f"Error loading image at idx {idx}: {e}")
            raise
