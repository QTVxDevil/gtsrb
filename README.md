# GTSRB & Zalo Traffic Sign Recognition

This project provides a full pipeline for training, fine-tuning, transfer learning, and evaluating deep learning models on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) and Zalo AI Challenge traffic sign datasets.

## Project Structure

```
├── base_train.py           # Base training script for GTSRB
├── fine-tune.py            # Fine-tuning script with STN and advanced scheduling
├── transfer_learning.py    # Transfer learning & testing for Zalo dataset
├── zalo_test.py            # Test and evaluation script for Zalo
├── test.py                 # Test and evaluation script for GTSRB
├── validate.py             # Simple validation script
├── demo_stn.py             # Visualize STN effects
├── dataloader/             # Data loading and augmentation
│   ├── gtsrb_dataloader.py
│   ├── zalo_dataloader.py
│   ├── transform.py
├── models/                 # Model definitions
│   ├── resnet50.py
│   ├── resnet_stn.py
│   ├── stn.py
├── src/                    # Configs and utilities
│   ├── cfg.py
│   ├── earlystoping.py
├── logs/                   # Model checkpoints (ignored by git)
├── figures/                # Plots and results (ignored by git)
├── datasets/               # GTSRB & Zalo datasets (ignored by git)
└── .gitignore
```

## Setup

1. **Install dependencies**
   ```bash
   pip install torch torchvision pandas scikit-learn matplotlib tqdm seaborn
   ```

2. **Download the GTSRB and/or Zalo datasets**
   - Place the extracted folders under `datasets/GTSRB_Traffic_Sign/` and `datasets/zalo_process/` as structured above.

3. **Configure paths and parameters**
   - Edit `src/cfg.py` if you want to change dataset paths, batch size, learning rate, etc.

## Usage

### GTSRB Training
Train a ResNet50 or ResNet50+STN model from scratch:
```bash
python base_train.py
```

### GTSRB Fine-tuning
Continue training with advanced scheduling and partial layer unfreezing:
```bash
python fine-tune.py
```

### GTSRB Validation
Evaluate the model on the GTSRB validation set:
```bash
python validate.py
```

### GTSRB Testing & Metrics
Run detailed evaluation, confusion matrix, and save predictions:
```bash
python test.py
```

### Zalo Transfer Learning & Testing
Transfer learning and test on the Zalo dataset:
```bash
python transfer_learning.py
python zalo_test.py
```

## Features
- Data augmentation and normalization (see `dataloader/transform.py`)
- Support for Spatial Transformer Networks (STN)
- Early stopping and learning rate scheduling
- Training/validation loss and accuracy plots
- Test set evaluation with precision, recall, F1-score, and confusion matrix
- Robust transfer learning from GTSRB to Zalo (automatic head replacement)
- Utility scripts for model inspection and reproducibility

## Checkpoints & Results
- Model checkpoints are saved in `logs/` (see `.gitignore`)
- Training/validation plots and confusion matrices are saved in `figures/`

## Notes
- The dataset and large files are ignored by git (see `.gitignore`).
- For best results, use a GPU-enabled environment.
- Zalo dataset must be preprocessed into class folders (1-7) under `datasets/zalo_process/`.

## License
This project is for academic and research purposes. See the GTSRB and Zalo dataset licenses for data usage terms.
