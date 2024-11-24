import torch
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
from lung_registration.utils.metrics import RegistrationMetrics

class RegistrationPredictor:
    """Class to handle model inference"""
    
    def __init__(self, model, device):
        """
        Initialize predictor
        Args:
            model: Registration network
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.device = device
        self.metrics = RegistrationMetrics()

    def predict_case(self, data):
        """Predict single case"""
        self.model.eval()
        with torch.no_grad():
            # Prepare data
            fixed = data['fixed'].unsqueeze(0).to(self.device)
            moving = data['moving'].unsqueeze(0).to(self.device)
            fixed_mask = data['fixed_mask'].unsqueeze(0).to(self.device)
            moving_mask = data['moving_mask'].unsqueeze(0).to(self.device)
            fixed_keypoints = data['fixed_keypoints'].to(self.device)
            moving_keypoints = data['moving_keypoints'].to(self.device)

            # Forward pass
            warped_moving, deformation = self.model(fixed, moving)
            warped_mask = self.model.warp(moving_mask, deformation)

            return {
                'warped_moving': warped_moving,
                'warped_mask': warped_mask,
                'deformation': deformation,
                'fixed': fixed,
                'moving': moving,
                'fixed_mask': fixed_mask,
                'moving_mask': moving_mask,
                'fixed_keypoints': fixed_keypoints,
                'moving_keypoints': moving_keypoints
            }

    def inference_on_dataset(self, dataset, save_dir: Path):
        """Run inference on entire dataset"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results = []
        
        for idx in tqdm(range(len(dataset)), desc="Processing test cases"):
            # Get prediction
            data = dataset[idx]
            prediction = self.predict_case(data)
            
            # Calculate metrics
            dsc = self.metrics.compute_dsc(prediction['warped_mask'], prediction['fixed_mask'])
            mse = self.metrics.compute_mse(prediction['warped_moving'], prediction['fixed'])
            ncc = self.metrics.compute_ncc(prediction['warped_moving'], prediction['fixed'])
            tre = self.metrics.compute_tre(prediction['moving_keypoints'], prediction['fixed_keypoints'])
            
            results.append({
                'case_id': idx + 1,
                'dsc': dsc,
                'mse': mse,
                'ncc': ncc,
                'mean_tre': tre['mean_tre'],
                'std_tre': tre['std_tre'],
                'max_tre': tre['max_tre']
            })

        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(save_dir / 'test_results.csv', index=False)
        
        return df_results