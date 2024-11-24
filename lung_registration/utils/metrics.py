from typing import Dict
import torch

class RegistrationMetrics:
    """Class for computing registration metrics"""
    
    @staticmethod
    def compute_dsc(pred_mask: torch.Tensor, target_mask: torch.Tensor, smooth: float = 1e-6) -> float:
        """Compute Dice Similarity Coefficient"""
        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask) + torch.sum(target_mask)
        dsc = (2. * intersection + smooth) / (union + smooth)
        return dsc.item()

    @staticmethod
    def compute_mse(pred_image: torch.Tensor, target_image: torch.Tensor) -> float:
        """Compute Mean Squared Error"""
        mse = torch.mean((pred_image - target_image) ** 2)
        return mse.item()

    @staticmethod
    def compute_ncc(pred_image: torch.Tensor, target_image: torch.Tensor) -> float:
        """Compute Normalized Cross Correlation"""
        pred_mean = torch.mean(pred_image)
        target_mean = torch.mean(target_image)
        pred_var = torch.mean((pred_image - pred_mean) ** 2)
        target_var = torch.mean((target_image - target_mean) ** 2)
        covariance = torch.mean((pred_image - pred_mean) * (target_image - target_mean))
        ncc = covariance / torch.sqrt(pred_var * target_var)
        return ncc.item()

    @staticmethod
    def compute_tre(pred_points: torch.Tensor, target_points: torch.Tensor) -> Dict[str, float]:
        """Compute Target Registration Error"""
        distances = torch.sqrt(((pred_points - target_points) ** 2).sum(dim=1))
        return {
            'mean_tre': distances.mean().item(),
            'std_tre': distances.std().item(),
            'max_tre': distances.max().item()
        }