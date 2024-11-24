from dataclasses import dataclass
from pathlib import Path
import torch
import os

@dataclass
class Config:
    """Global configuration class"""
    # Data paths
    base_dir: str = str(Path(__file__).parent.parent.parent / "LungCT") 
    
    # Dataset settings
    train_start_idx: int = 1
    train_end_idx: int = 20
    test_start_idx: int = 21
    test_end_idx: int = 30
    
    # Model parameters
    in_channels: int = 2
    out_channels: int = 3
    channels: tuple = (16, 32, 64, 128)
    strides: tuple = (2, 2, 2)
    num_res_units: int = 2
    
    # Training parameters
    batch_size: int = 2
    num_epochs: int = 500
    learning_rate: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loss weights
    similarity_weight: float = 1.0
    smoothness_weight: float = 0.1
    dice_weight: float = 1.0
    keypoint_weight: float = 0.5
    
    # Augmentation
    augment_data: bool = True
    
    def __post_init__(self):
        """Initialize paths and create directories"""
        # Convert base_dir to Path object
        self.base_dir = Path(self.base_dir)
        
        # Create results directory
        self.results_dir = self.base_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Create checkpoints directory
        self.checkpoints_dir = self.base_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Best model path
        self.model_path = self.checkpoints_dir / 'best_lung_registration_model.pth'
        
        # Verify data directories exist
        required_folders = ['imagesTr', 'imagesTs', 'masksTr', 'masksTs', 
                          'keypointsTr', 'keypointsTs']
        for folder in required_folders:
            if not (self.base_dir / folder).exists():
                raise ValueError(f"Required folder missing: {folder}")
    
    @property
    def device_obj(self) -> torch.device:
        """Get PyTorch device object"""
        return torch.device(self.device)

# Create default configuration
config = Config()