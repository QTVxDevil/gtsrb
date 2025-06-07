import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dataloader.transform import get_transform
from dataloader.gtsrb_dataloader import GTSRB_load
from src.cfg import GTSRB_TRAINING_PATH, RESNET_FIGURE_PATH, NORMALIZE_PARAMETER
import torch
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        unnormalized_tensor = (tensor * self.std) + self.mean
        return unnormalized_tensor

MEAN = NORMALIZE_PARAMETER['mean']
STD = NORMALIZE_PARAMETER['std']

transform_dict = {
    'Original (Un-normalized Display)': transforms.Compose([
        UnNormalize(mean=MEAN, std=STD), 
        transforms.ToPILImage(),         
    ]),
    'Resize': transforms.Compose([
        transforms.Resize((48, 48)),
    ]),
    'RandomRotation': transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomRotation(15),
    ]),
    'ColorJitter': transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
    ]),
    'GaussianBlur': transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]),
    'FullTrainTransform': get_transform(train=True),
    'FullValTransform': get_transform(train=False),
}

def show_images(imgs, titles, suptitle, save_path=None):
    plt.figure(figsize=(15, 3))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, len(imgs), i+1)
        if isinstance(img, (Image.Image,)):
            plt.imshow(img)
        elif isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
        else:
            print(f"Warning: Image type not handled for display: {type(img)}")
            plt.imshow(img) 
        plt.title(title)
        plt.axis('off')
    plt.suptitle(suptitle)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    train_dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='train')
    indices = random.sample(range(len(train_dataset)), 5)

    pre_transformed_and_normalized_tensors = []
    raw_pil_images_for_individual_transforms = []
    img_names = []

    for i in indices:
        sample_tensor, _ = train_dataset[i]
        pre_transformed_and_normalized_tensors.append(sample_tensor)
        img_names.append(f"train_{i}")

        unnormalized_tensor = UnNormalize(mean=MEAN, std=STD)(sample_tensor)
        pil_img = transforms.ToPILImage()(unnormalized_tensor)
        raw_pil_images_for_individual_transforms.append(pil_img)
    
    if not os.path.exists(RESNET_FIGURE_PATH):
        os.makedirs(RESNET_FIGURE_PATH)

    for tname, tform in transform_dict.items():
        transformed_imgs = []
        
        if tname == 'Original (Un-normalized Display)':
            for img_tensor in pre_transformed_and_normalized_tensors:
                pil_img = tform(img_tensor) 
                transformed_imgs.append(pil_img)
            
            show_images(transformed_imgs, img_names,
                        f"Transform: {tname}",
                        save_path=os.path.join(RESNET_FIGURE_PATH, f"transform_{tname.replace(' ', '_').replace('(', '').replace(')', '')}.png"))
        
        elif tname in ['FullTrainTransform', 'FullValTransform']:
            save_path = os.path.join(RESNET_FIGURE_PATH, f"transform_{tname}.png")
            show_images(pre_transformed_and_normalized_tensors, img_names,
                        f"Transform: {tname} (Dataset Output)", save_path=save_path)
        
        else:
            for pil_img_source in raw_pil_images_for_individual_transforms:
                transformed_pil_img = tform(pil_img_source)
                transformed_imgs.append(transformed_pil_img)
            save_path = os.path.join(RESNET_FIGURE_PATH, f"transform_{tname}.png")
            show_images(transformed_imgs, img_names, f"Transform: {tname}", save_path=save_path)

if __name__ == '__main__':
    main()