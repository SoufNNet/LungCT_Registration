from monai.transforms import (
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    Compose
)
from typing import Optional

def get_transforms(augment: bool = False) -> Optional[Compose]:
    """Get transforms for data augmentation
    
    Args:
        augment: Whether to return augmentation transforms or None
    
    Returns:
        Composition of transforms if augment=True, None otherwise
    """
    if not augment:
        return None
        
    return Compose([
        RandFlipd(
            keys=["fixed", "moving", "fixed_mask", "moving_mask"],  # Added masks to the transforms
            prob=0.5,
            spatial_axis=0
        ),
        RandRotate90d(
            keys=["fixed", "moving", "fixed_mask", "moving_mask"],  # Added masks to the transforms
            prob=0.5,
            max_k=3
        ),
        RandZoomd(
            keys=["fixed", "moving", "fixed_mask", "moving_mask"],  # Added masks to the transforms
            prob=0.5,
            min_zoom=0.8,
            max_zoom=1.2,
            mode=("bilinear", "bilinear", "nearest", "nearest")  # Different interpolation for images and masks
        )
    ])