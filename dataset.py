from torch.utils.data import Dataset
import pathlib
from PIL import Image
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        """
        Custom dataset for image segmentation.
        
        Args:
            images_dir: Path to directory containing original images
            masks_dir: Path to directory containing ground truth masks
            image_transform: Transformations to apply to images
            mask_transform: Transformations to apply to masks
        """
        self.images_dir = pathlib.Path(images_dir)
        self.masks_dir = pathlib.Path(masks_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        # Get list of image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Verify that corresponding mask files exist
        self.valid_pairs = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / f"{img_file.stem}_Segmentation.png"
            if mask_file.exists():
                self.valid_pairs.append((img_file, mask_file))
            else:
                print(f"Warning: No mask found for {img_file.name}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load as grayscale
        
        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

def custom_collate_fn(batch):
    """Custom collate function to handle any remaining size inconsistencies"""
    images, masks = zip(*batch)
    
    # Stack images and masks
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return images, masks


def get_dataloaders(dataset, batch_size=8, shuffle=True, num_workers=0, t_size=0.7, v_size=0.2, eval=True):
    train_size = int(t_size * len(dataset))
    if eval:
        val_size = int(v_size * len(dataset))
        eval_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, eval_dataset = random_split(
            dataset, [train_size, val_size, eval_size]
        )
    else:
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    # Create separate dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    if eval:
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=1,  # Single images for visualization
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=custom_collate_fn
        )
        return train_loader, val_loader, eval_loader
    else:
        return train_loader, val_loader