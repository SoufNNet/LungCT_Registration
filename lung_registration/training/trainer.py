import torch
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
from lung_registration.models.network import RegistrationNetwork
from lung_registration.models.losses import RegistrationLoss
from lung_registration.utils.metrics import RegistrationMetrics

class RegistrationTrainer:
    """Class to handle the training of the registration model"""
    
    def __init__(self, model: RegistrationNetwork, device: torch.device, lr: float = 1e-3):
        """
        Initialize trainer
        Args:
            model: Registration network
            device: Device to train on
            lr: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = RegistrationLoss()
        self.metrics = RegistrationMetrics()

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'loss': 0, 'dsc': 0, 'mse': 0, 'ncc': 0,
            'tre_mean': 0, 'tre_std': 0, 'tre_max': 0
        }
        progress = tqdm(dataloader, desc='Training')

        for batch in progress:
            # Get data
            fixed = batch['fixed'].to(self.device)
            moving = batch['moving'].to(self.device)
            fixed_mask = batch['fixed_mask'].to(self.device)
            moving_mask = batch['moving_mask'].to(self.device)
            fixed_keypoints = batch['fixed_keypoints'].to(self.device)
            moving_keypoints = batch['moving_keypoints'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            warped_moving, deformation = self.model(fixed, moving)
            warped_mask = self.model.warp(moving_mask, deformation)

            # Calculate loss
            loss, loss_dict = self.criterion(
                warped_moving, fixed,
                warped_mask, fixed_mask,
                moving_keypoints, fixed_keypoints,
                deformation
            )

            # Calculate metrics
            dsc = self.metrics.compute_dsc(warped_mask, fixed_mask)
            mse = self.metrics.compute_mse(warped_moving, fixed)
            ncc = self.metrics.compute_ncc(warped_moving, fixed)
            tre_metrics = self.metrics.compute_tre(
                self.criterion.keypoint_loss.transform_keypoints(moving_keypoints, deformation),
                fixed_keypoints
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['dsc'] += dsc
            epoch_metrics['mse'] += mse
            epoch_metrics['ncc'] += ncc
            epoch_metrics['tre_mean'] += tre_metrics['mean_tre']
            epoch_metrics['tre_std'] += tre_metrics['std_tre']
            epoch_metrics['tre_max'] += tre_metrics['max_tre']

            progress.set_postfix({
                'loss': loss.item(),
                'DSC': dsc,
                'TRE': tre_metrics['mean_tre']
            })

        # Average metrics
        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        epoch_metrics = {
            'loss': 0, 'dsc': 0, 'mse': 0, 'ncc': 0,
            'tre_mean': 0, 'tre_std': 0, 'tre_max': 0
        }
        progress = tqdm(dataloader, desc='Validation')

        for batch in progress:
            # Process batch (same as training but without gradient)
            fixed = batch['fixed'].to(self.device)
            moving = batch['moving'].to(self.device)
            fixed_mask = batch['fixed_mask'].to(self.device)
            moving_mask = batch['moving_mask'].to(self.device)
            fixed_keypoints = batch['fixed_keypoints'].to(self.device)
            moving_keypoints = batch['moving_keypoints'].to(self.device)

            warped_moving, deformation = self.model(fixed, moving)
            warped_mask = self.model.warp(moving_mask, deformation)

            loss, _ = self.criterion(
                warped_moving, fixed,
                warped_mask, fixed_mask,
                moving_keypoints, fixed_keypoints,
                deformation
            )

            # Calculate metrics
            dsc = self.metrics.compute_dsc(warped_mask, fixed_mask)
            mse = self.metrics.compute_mse(warped_moving, fixed)
            ncc = self.metrics.compute_ncc(warped_moving, fixed)
            tre_metrics = self.metrics.compute_tre(
                self.criterion.keypoint_loss.transform_keypoints(moving_keypoints, deformation),
                fixed_keypoints
            )

            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['dsc'] += dsc
            epoch_metrics['mse'] += mse
            epoch_metrics['ncc'] += ncc
            epoch_metrics['tre_mean'] += tre_metrics['mean_tre']
            epoch_metrics['tre_std'] += tre_metrics['std_tre']
            epoch_metrics['tre_max'] += tre_metrics['max_tre']

            progress.set_postfix({
                'loss': loss.item(),
                'DSC': dsc,
                'TRE': tre_metrics['mean_tre']
            })

        # Average metrics
        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)