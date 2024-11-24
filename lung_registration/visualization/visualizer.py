import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np

class RegistrationVisualizer:
    """Class to handle visualization of registration results"""
    
    def __init__(self, save_dir: Path):
        """
        Initialize visualizer
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def visualize_case(self, prediction: dict, case_id: int):
        """Visualize single case results
        Args:
            prediction: Dictionary containing registration results
            case_id: Case identifier
        """
        # Convert tensors to CPU if they're not already
        prediction = {k: v.cpu() if torch.is_tensor(v) and v.device.type != 'cpu' else v 
                     for k, v in prediction.items()}

        # Print shapes for debugging
        print("\nData shapes for visualization:")
        for key, value in prediction.items():
            if torch.is_tensor(value):
                print(f"{key} shape: {value.shape}")

        # Get middle slice for visualization
        slice_idx = prediction['fixed'].shape[2] // 2

        # Create figure
        fig = plt.figure(figsize=(20, 10))

        # Fixed Image
        plt.subplot(231)
        plt.imshow(prediction['fixed'][0, 0, slice_idx], cmap='gray')
        plt.contour(prediction['fixed_mask'][0, 0, slice_idx], colors='r', levels=[0.5])
        plt.title('Fixed Image')
        plt.axis('off')

        # Moving Image
        plt.subplot(232)
        plt.imshow(prediction['moving'][0, 0, slice_idx], cmap='gray')
        plt.contour(prediction['moving_mask'][0, 0, slice_idx], colors='r', levels=[0.5])
        plt.title('Moving Image')
        plt.axis('off')

        # Warped Image
        plt.subplot(233)
        plt.imshow(prediction['warped_moving'][0, 0, slice_idx], cmap='gray')
        plt.contour(prediction['warped_mask'][0, 0, slice_idx], colors='r', levels=[0.5])
        plt.title('Registered Image')
        plt.axis('off')

        # Deformation field components
        for i in range(3):
            plt.subplot(234 + i)
            plt.imshow(prediction['deformation'][0, i, slice_idx])
            plt.title(f'Deformation Field Component {i+1}')
            plt.colorbar()
            plt.axis('off')

        # Save the figure
        plt.tight_layout()
        save_path = self.save_dir / f'registration_case_{case_id}.png'
        print(f"\nSaving visualization to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Save individual view plots (axial, coronal, sagittal)
        views = {
            'Axial': (slice_idx, lambda x: x[0, 0, slice_idx]),
            'Coronal': (prediction['fixed'].shape[3] // 2, lambda x: x[0, 0, :, prediction['fixed'].shape[3] // 2, :]),
            'Sagittal': (prediction['fixed'].shape[4] // 2, lambda x: x[0, 0, :, :, prediction['fixed'].shape[4] // 2])
        }

        for view_name, (slice_idx, slice_func) in views.items():
            fig = plt.figure(figsize=(20, 5))

            # Fixed Image
            plt.subplot(131)
            plt.imshow(slice_func(prediction['fixed']), cmap='gray')
            plt.contour(slice_func(prediction['fixed_mask']), colors='r', levels=[0.5])
            plt.title(f'Fixed Image ({view_name})')
            plt.axis('off')

            # Moving Image
            plt.subplot(132)
            plt.imshow(slice_func(prediction['moving']), cmap='gray')
            plt.contour(slice_func(prediction['moving_mask']), colors='r', levels=[0.5])
            plt.title(f'Moving Image ({view_name})')
            plt.axis('off')

            # Warped Image
            plt.subplot(133)
            plt.imshow(slice_func(prediction['warped_moving']), cmap='gray')
            plt.contour(slice_func(prediction['warped_mask']), colors='r', levels=[0.5])
            plt.title(f'Warped Image ({view_name})')
            plt.axis('off')

            plt.tight_layout()
            save_path = self.save_dir / f'registration_case_{case_id}_{view_name.lower()}.png'
            print(f"Saving {view_name} view to: {save_path}")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Save deformation field visualization for each view
            plt.figure(figsize=(15, 5))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                deformation_slice = slice_func(prediction['deformation'][:, i:i+1])
                plt.imshow(deformation_slice, cmap='rainbow')
                plt.title(f'Deformation {i+1} ({view_name})')
                plt.colorbar()
                plt.axis('off')

            plt.tight_layout()
            save_path = self.save_dir / f'deformation_case_{case_id}_{view_name.lower()}.png'
            print(f"Saving {view_name} deformation to: {save_path}")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

    def plot_training_curves(self, metrics_history: dict):
        """Plot training metrics over time"""
        plt.figure(figsize=(15, 5))
        metrics_to_plot = ['loss', 'dsc', 'tre_mean']
        titles = ['Loss', 'Dice Score', 'Target Registration Error']
        
        for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            plt.subplot(1, 3, i+1)
            plt.plot(metrics_history['train'][metric], label='Train')
            plt.plot(metrics_history['val'][metric], label='Val')
            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.legend()
        
        plt.tight_layout()
        save_path = self.save_dir / 'training_curves.png'
        print(f"\nSaving training curves to: {save_path}")
        plt.savefig(save_path)
        plt.close()