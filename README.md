# GTSRB Traffic Sign Recognition

This project provides a full pipeline for training, fine-tuning, and evaluating deep learning models on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) dataset.

## Project Structure

```
├── base_train.py           # Base training script
├── fine-tune.py            # Fine-tuning script with STN and advanced scheduling
├── test.py                 # Test and evaluation script with detailed metrics
├── validate.py             # Simple validation script
├── dataloader/             # Data loading and augmentation
│   ├── gtsrb_dataloader.py
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
├── datasets/               # GTSRB dataset (ignored by git)
└── .gitignore
```

## Setup

1. **Install dependencies**
   ```bash
   pip install torch torchvision pandas scikit-learn matplotlib tqdm seaborn
   ```

2. **Download the GTSRB dataset**
   - Place the extracted folders under `datasets/GTSRB_Traffic_Sign/` as structured above.

3. **Configure paths and parameters**
   - Edit `src/cfg.py` if you want to change dataset paths, batch size, learning rate, etc.

## Usage

### Training
Train a ResNet50 or ResNet50+STN model from scratch:
```bash
python base_train.py
```

### Fine-tuning
Continue training with advanced scheduling and partial layer unfreezing:
```bash
python fine-tune.py
```

### Validation
Evaluate the model on the test set:
```bash
python validate.py
```

### Testing & Metrics
Run detailed evaluation, confusion matrix, and save predictions:
```bash
python test.py
```

## Features
- Data augmentation and normalization (see `dataloader/transform.py`)
- Support for Spatial Transformer Networks (STN)
- Early stopping and learning rate scheduling
- Training/validation loss and accuracy plots
- Test set evaluation with precision, recall, F1-score, and confusion matrix

## Checkpoints & Results
- Model checkpoints are saved in `logs/` (see `.gitignore`)
- Training/validation plots and confusion matrices are saved in `figures/`

## Notes
- The dataset and large files are ignored by git (see `.gitignore`).
- For best results, use a GPU-enabled environment.

## License
This project is for academic and research purposes. See the GTSRB dataset license for data usage terms.
