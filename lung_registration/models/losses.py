import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import BendingEnergyLoss, DiceLoss
from monai.losses import GlobalMutualInformationLoss

class KeypointLoss(nn.Module):
    """Loss based on keypoint correspondence"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, moving_points: torch.Tensor, 
                fixed_points: torch.Tensor, 
                deformation_field: torch.Tensor) -> torch.Tensor:
        """Calculate keypoint loss

        Args:
            moving_points: Moving keypoints
            fixed_points: Fixed keypoints
            deformation_field: Deformation field
            
        Returns:
            Keypoint loss value
        """
        # Transform keypoints using deformation field
        warped_points = self.transform_keypoints(moving_points, deformation_field)
        # Calculate mean squared error between warped and fixed points
        loss = torch.mean((warped_points - fixed_points) ** 2)
        return loss
        
    def transform_keypoints(self, keypoints: torch.Tensor, 
                          deformation_field: torch.Tensor) -> torch.Tensor:
        """Transform keypoints using deformation field"""
        device = deformation_field.device
        keypoints = keypoints.to(device)
        
        # Normalize coordinates to [-1, 1]
        size = torch.tensor(deformation_field.shape[2:], device=device)
        normalized_points = 2.0 * keypoints / (size - 1) - 1.0
        
        # Sample deformation field at keypoint locations
        deformations = F.grid_sample(
            deformation_field,
            normalized_points.view(1, 1, -1, 1, 3),
            mode='bilinear',
            align_corners=True
        )
        
        # Apply deformation
        transformed_points = keypoints + deformations.squeeze().t()
        return transformed_points

class RegistrationLoss:
    """Combined loss for registration"""
    
    def __init__(self):
        """Initialize all loss components"""
        self.similarity_loss = GlobalMutualInformationLoss()
        self.smoothness_loss = BendingEnergyLoss()
        self.dice_loss = DiceLoss()
        self.keypoint_loss = KeypointLoss()
        
        # Loss weights from original code
        self.weights = {
            'similarity': 1.0,
            'smoothness': 0.1,
            'dice': 1.0,
            'keypoint': 0.5
        }
        
    def __call__(self, 
                 warped_moving: torch.Tensor,
                 fixed: torch.Tensor,
                 warped_mask: torch.Tensor,
                 fixed_mask: torch.Tensor,
                 moving_keypoints: torch.Tensor,
                 fixed_keypoints: torch.Tensor,
                 deformation: torch.Tensor) -> tuple:
        """Calculate total loss
        
        Args:
            warped_moving: Warped moving image
            fixed: Fixed image
            warped_mask: Warped moving mask
            fixed_mask: Fixed mask
            moving_keypoints: Moving keypoints
            fixed_keypoints: Fixed keypoints
            deformation: Deformation field
            
        Returns:
            Tuple of (total loss, dictionary of individual losses)
        """
        # Calculate individual losses
        sim_loss = self.similarity_loss(warped_moving, fixed)
        smooth_loss = self.smoothness_loss(deformation)
        dice_loss = self.dice_loss(warped_mask, fixed_mask)
        kp_loss = self.keypoint_loss(moving_keypoints, fixed_keypoints, deformation)

        # Calculate weighted total loss
        total_loss = (
            self.weights['similarity'] * sim_loss +
            self.weights['smoothness'] * smooth_loss +
            self.weights['dice'] * dice_loss +
            self.weights['keypoint'] * kp_loss
        )

        # Return individual losses for logging
        losses_dict = {
            'total': total_loss.item(),
            'similarity': sim_loss.item(),
            'smoothness': smooth_loss.item(),
            'dice': dice_loss.item(),
            'keypoint': kp_loss.item()
        }
        
        return total_loss, losses_dict