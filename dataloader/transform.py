from torchvision import transforms
from src.cfg import NORMALIZE_PARAMETER, IMAGE_SIZE

def get_transform(train=True):
    target_image_size = (IMAGE_SIZE, IMAGE_SIZE) if isinstance(IMAGE_SIZE, int) else IMAGE_SIZE


    if train:
        transform = transforms.Compose([
            transforms.Resize(target_image_size),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomCrop(target_image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_PARAMETER['mean'],
                                 std=NORMALIZE_PARAMETER['std']),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(target_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_PARAMETER['mean'],
                                 std=NORMALIZE_PARAMETER['std']),
        ])
    return transform