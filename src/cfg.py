import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTSRB_BASE_DIR = os.path.join(BASE_DIR, 'datasets/GTSRB_Traffic_Sign')

GTSRB_TRAINING_PATH = os.path.join(GTSRB_BASE_DIR, 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')
GTSRB_TEST_PATH = os.path.join(GTSRB_BASE_DIR, 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')

RESNET_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'logs/resnet50_1.pth')
RESNET_CHECKPOINT_PATH_2 = os.path.join(BASE_DIR, 'logs/resnet50_2.pth')
RESNET_FIGURE_PATH = os.path.join(BASE_DIR, 'figures')
GTSRB_CLASS_WEIGHT_PATH = os.path.join(BASE_DIR, 'logs/gtsrb_class_weights.npy')

IMAGE_SIZE = (48, 48)
GTSRB_NUM_CLASSES = 43

BATCH = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
HARD_MINING_FREQ = 5

NORMALIZE_PARAMETER = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EARLY_STOPPING_PARAMS = {
    'patience': 8,
    'delta': 0.0,
    'verbose': True
}