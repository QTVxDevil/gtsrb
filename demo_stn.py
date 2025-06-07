import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataloader.gtsrb_dataloader import GTSRB_load
from models.resnet_stn import ResNetWithSTN
from src.cfg import (GTSRB_TRAINING_PATH, IMAGE_SIZE, GTSRB_NUM_CLASSES,
                     DEVICE, RESNET_CHECKPOINT_PATH_4, NORMALIZE_PARAMETER)
from torchvision import transforms
import os
import numpy as np

def unnormalize_image(tensor):
    mean = torch.tensor(NORMALIZE_PARAMETER['mean']).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(NORMALIZE_PARAMETER['std']).view(1, 3, 1, 1).to(tensor.device)
    unnormalized_tensor = tensor * std + mean

    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    return unnormalized_tensor

def visualize_stn_transformation(model, original_images_batch, titles=None, save_path=None):
    model.eval()
    
    with torch.no_grad():
        x_device = original_images_batch.to(DEVICE)
        
        x = model.conv1_modified(x_device)
        x = model.bn1_modified(x)
        x = model.relu_modified(x)
        x_pre_stn = model.maxpool_modified(x)

        theta = model.stn.localization_net(x_pre_stn)

        grid = F.affine_grid(theta, original_images_batch.size(), align_corners=False)
        transformed_images_for_viz = F.grid_sample(original_images_batch.to(DEVICE), grid, align_corners=False)

    original_images_display = unnormalize_image(original_images_batch.cpu())
    transformed_images_display = unnormalize_image(transformed_images_for_viz.cpu())

    n = original_images_display.size(0)

    plt.figure(figsize=(n * 3, 6))

    for i in range(n):
        plt.subplot(2, n, i + 1)
        img_np = original_images_display[i].permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.axis('off')

        plt.subplot(2, n, n + i + 1)
        timg_np = transformed_images_display[i].permute(1, 2, 0).numpy()
        plt.imshow(timg_np)
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def demo_stn_on_training_images_trained_model():
    print(f"Using device: {DEVICE}")

    model = ResNetWithSTN(num_classes=GTSRB_NUM_CLASSES, stn_filters=(16, 32),
                          stn_fc_units=128, input_size=IMAGE_SIZE)

    def load_checkpoint(model, checkpoint_path, device):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded model weights from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading model state_dict from {checkpoint_path}: {e}")
                print("This usually happens if module names in checkpoint do not exactly match model architecture.")
                print("Attempting to load with partial matches...")
                model_state_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
                model_state_dict.update(pretrained_dict)
                model.load_state_dict(model_state_dict, strict=False)
                print("Loaded partial model weights (mismatched/missing keys were skipped).")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting with pretrained ImageNet weights for ResNet backbone (if applicable).")
            print("WARNING: This might lead to sub-optimal results if this is not the first stage and no checkpoint was loaded.")

    try:
        load_checkpoint(model, RESNET_CHECKPOINT_PATH_4, DEVICE)
    except RuntimeError as e:
        print(f"Could not load checkpoint {RESNET_CHECKPOINT_PATH_4}. Please ensure your model is trained and this path is correct. Error: {e}")
        print("Exiting demo as trained model weights are required.")
        return

    model.to(DEVICE)
    print("Trained ResNetWithSTN model loaded successfully.")

    dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='val') 

    num_images_to_show = 5

    selected_indices = [
        np.random.randint(0, len(dataset)) for _ in range(num_images_to_show)
    ]
    
    sample_images = []
    titles = []
    for i, idx in enumerate(selected_indices):
        img_tensor, label = dataset[idx] 
        sample_images.append(img_tensor)
    
    batch = torch.stack(sample_images)

    print(f"Visualizing STN effect on {num_images_to_show} sample images...")
    visualize_stn_transformation(model, batch, titles=titles, save_path="figures/stn_trained_model_effect_demo.png")
    print("Visualization complete. Check 'figures/stn_trained_model_effect_demo.png'")


if __name__ == '__main__':
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    demo_stn_on_training_images_trained_model()