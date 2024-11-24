import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.blocks import Warp

class RegistrationNetwork(nn.Module):
    """Registration network based on UNet architecture"""
    
    def __init__(self):
        """Initialize the network"""
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        )
        self.warp = Warp(mode='bilinear', padding_mode='zeros')

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> tuple:
        """Forward pass
        
        Args:
            fixed: Fixed image
            moving: Moving image
            
        Returns:
            Tuple of (warped moving image, deformation field)
        """
        x = torch.cat([fixed, moving], dim=1)
        deformation = self.net(x)
        warped_moving = self.warp(moving, deformation)
        return warped_moving, deformation