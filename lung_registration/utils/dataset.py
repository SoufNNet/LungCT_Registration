from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
from lung_registration.configs.config import Config
from lung_registration.utils.transforms import get_transforms

class LungDataset(Dataset):
    """Dataset class for lung CT image pairs with detailed statistics"""
    
    def __init__(self, config: Config, phase: str = 'train', augment: Optional[bool] = None):
        """Initialize dataset
        
        Args:
            config: Configuration object
            phase: 'train' or 'test'
            augment: Whether to apply augmentation (defaults to config value)
        """
        self.config = config
        self.phase = phase
        self.augment = config.augment_data if augment is None else augment
        
        # Set phase-specific parameters
        if phase == 'train':
            self.start_idx = config.train_start_idx
            self.end_idx = config.train_end_idx
            img_folder = 'imagesTr'
            mask_folder = 'masksTr'
            keypoints_folder = 'keypointsTr'
        else:  # test phase
            self.start_idx = config.test_start_idx
            self.end_idx = config.test_end_idx
            img_folder = 'imagesTs'
            mask_folder = 'masksTs'
            keypoints_folder = 'keypointsTs'

        # Initialize augmentation transforms
        self.transforms = get_transforms(self.augment)

        # Create data pairs with validation
        self.pairs = []
        self.available_pairs = []
        self.missing_files = []
        
        for i in range(self.start_idx, self.end_idx + 1):
            idx = str(i).zfill(4)
            pair = {
                'fixed': config.base_dir / img_folder / f'LungCT_{idx}_0000.nii.gz',
                'moving': config.base_dir / img_folder / f'LungCT_{idx}_0001.nii.gz',
                'fixed_mask': config.base_dir / mask_folder / f'LungCT_{idx}_0000.nii.gz',
                'moving_mask': config.base_dir / mask_folder / f'LungCT_{idx}_0001.nii.gz',
                'fixed_keypoints': config.base_dir / keypoints_folder / f'LungCT_{idx}_0000.csv',
                'moving_keypoints': config.base_dir / keypoints_folder / f'LungCT_{idx}_0001.csv'
            }
            
            # Check if all files exist
            missing = [k for k, v in pair.items() if not v.exists()]
            if not missing:
                self.pairs.append(pair)
                self.available_pairs.append(idx)
            else:
                self.missing_files.append({
                    'patient_id': idx,
                    'missing': missing
                })

        if len(self.pairs) == 0:
            raise ValueError(f"No valid data pairs found for {phase} phase in {config.base_dir}")

    def load_and_normalize(self, path: Path, is_mask: bool = False) -> torch.Tensor:
        """Load and normalize image data
        
        Args:
            path: Path to the image file
            is_mask: Whether the image is a binary mask
            
        Returns:
            Normalized tensor
        """
        img = nib.load(str(path)).get_fdata()
        
        if not is_mask:
            img = np.clip(img, -1024, 3000)
            img = (img + 1024) / (3000 + 1024)
        else:
            img = (img > 0).astype(np.float32)
            
        return torch.from_numpy(img).float()

    def load_keypoints(self, path: Path) -> Optional[torch.Tensor]:
        """Load keypoints from CSV file
        
        Args:
            path: Path to the keypoints file
            
        Returns:
            Tensor of keypoint coordinates or None if loading fails
        """
        try:
            keypoints = pd.read_csv(path).values
            return torch.tensor(keypoints, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading keypoints {path}: {e}")
            return None

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the dataset"""
        first_item = self[0] if len(self) > 0 else None
        suffix = "Tr" if self.phase == "train" else "Ts"
        
        stats = {
            'phase': self.phase,
            'num_pairs': len(self.pairs),
            'augmentation_enabled': self.augment,
            'patient_ids': self.available_pairs,
            'missing_files': self.missing_files if self.missing_files else None,
            'index_range': f"{self.start_idx:04d}-{self.end_idx:04d}",
            'directories': {
                'images': str(self.config.base_dir / f'images{suffix}'),
                'masks': str(self.config.base_dir / f'masks{suffix}'),
                'keypoints': str(self.config.base_dir / f'keypoints{suffix}')
            }
        }
        
        if first_item is not None:
            stats['data_dimensions'] = {
                'fixed_image': tuple(first_item['fixed'].shape),
                'moving_image': tuple(first_item['moving'].shape),
                'fixed_mask': tuple(first_item['fixed_mask'].shape),
                'moving_mask': tuple(first_item['moving_mask'].shape)
            }
            if first_item['fixed_keypoints'] is not None:
                stats['keypoints_per_image'] = len(first_item['fixed_keypoints'])
        
        return stats

    @staticmethod
    def display_model_data_stats(train_dataset: 'LungDataset', 
                               test_dataset: 'LungDataset', 
                               val_dataset: Optional['LungDataset'] = None):
        """Display comprehensive statistics about model data"""
        datasets = {
            'Training': train_dataset,
            'Testing': test_dataset,
            'Validation': val_dataset
        }
        
        print("\n" + "="*80)
        print("MODEL INPUT DATA STATISTICS")
        print("="*80)
        
        total_pairs = 0
        
        for name, dataset in datasets.items():
            if dataset is None:
                continue
                
            stats = dataset.get_dataset_stats()
            color = '\033[92m' if name == 'Training' else '\033[94m' if name == 'Testing' else '\033[93m'
            reset_color = '\033[0m'
            
            total_pairs += stats['num_pairs']
            
            print(f"\n{color}{name} Set{reset_color}")
            print("-" * 40)
            print(f"Number of image pairs: {stats['num_pairs']}")
            print(f"Index range: {stats['index_range']}")
            print(f"Data augmentation: {'Enabled' if stats['augmentation_enabled'] else 'Disabled'}")
            
            if 'data_dimensions' in stats:
                print("\nInput Dimensions:")
                for key, dim in stats['data_dimensions'].items():
                    print(f"  {key}: {dim}")
                if 'keypoints_per_image' in stats:
                    print(f"  Keypoints per image: {stats['keypoints_per_image']}")
            
            print("\nDirectories:")
            for data_type, directory in stats['directories'].items():
                print(f"  {data_type}: {directory}")
            
            print("\nPatient IDs:", ', '.join(stats['patient_ids']))
            
            if stats.get('missing_files'):
                print("\nMissing Files:")
                for missing in stats['missing_files']:
                    print(f"  Patient {missing['patient_id']}: {', '.join(missing['missing'])}")
        
        print("\n" + "-"*40)
        print(f"Total Statistics:")
        print(f"Total number of pairs: {total_pairs}")
        print("="*80 + "\n")

    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data pair
        
        Args:
            idx: Index of the data pair
            
        Returns:
            Dictionary containing the image pair and associated data
        """
        pair = self.pairs[idx]

        # Load all data
        data = {
            'fixed': self.load_and_normalize(pair['fixed']).unsqueeze(0),
            'moving': self.load_and_normalize(pair['moving']).unsqueeze(0),
            'fixed_mask': self.load_and_normalize(pair['fixed_mask'], is_mask=True).unsqueeze(0),
            'moving_mask': self.load_and_normalize(pair['moving_mask'], is_mask=True).unsqueeze(0),
            'fixed_keypoints': self.load_keypoints(pair['fixed_keypoints']),
            'moving_keypoints': self.load_keypoints(pair['moving_keypoints'])
        }

        # Apply augmentation if enabled
        if self.augment and self.transforms is not None:
            data = self.transforms(data)

        return data