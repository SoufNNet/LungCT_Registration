import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from lung_registration.utils.dataset import LungDataset
from lung_registration.models.network import RegistrationNetwork
from lung_registration.training.trainer import RegistrationTrainer
from lung_registration.inference.predictor import RegistrationPredictor
from lung_registration.visualization.visualizer import RegistrationVisualizer
from lung_registration.configs.config import Config

def main():
    """Main training and inference pipeline"""
    print("Initializing registration pipeline...")
    
    # Initialize configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = LungDataset(config, phase='train')  
    val_dataset = LungDataset(config, phase='test')     
    
    # Display detailed dataset statistics
    LungDataset.display_model_data_stats(
        train_dataset=train_dataset,
        test_dataset=val_dataset
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Initialize model and trainer
    model = RegistrationNetwork()
    trainer = RegistrationTrainer(model, device, lr=config.learning_rate)
    
    # Initialize metrics tracking
    best_val_metrics = {'loss': float('inf')}
    metrics_history = {
        'train': {'loss': [], 'dsc': [], 'tre_mean': []},
        'val': {'loss': [], 'dsc': [], 'tre_mean': []}
    }
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"DSC: {train_metrics['dsc']:.4f}, "
              f"TRE: {train_metrics['tre_mean']:.4f} ± {train_metrics['tre_std']:.4f}")
        
        # Validation phase
        val_metrics = trainer.validate(val_loader)
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"DSC: {val_metrics['dsc']:.4f}, "
              f"TRE: {val_metrics['tre_mean']:.4f} ± {val_metrics['tre_std']:.4f}")
        
        # Update metrics history
        for phase in ['train', 'val']:
            metrics = train_metrics if phase == 'train' else val_metrics
            for key in ['loss', 'dsc', 'tre_mean']:
                metrics_history[phase][key].append(metrics[key])
        
        # Save best model
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            print("Saving best model...")
            trainer.save_checkpoint(
                config.model_path,
                epoch,
                val_metrics
            )
    
    # Load best model for inference
    print("\nLoading best model for inference...")
    checkpoint = torch.load(config.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize predictor and visualizer
    predictor = RegistrationPredictor(model, device)
    visualizer = RegistrationVisualizer(config.results_dir)
    
    # Run inference on validation set
    print("\nRunning inference on validation set...")
    results_df = predictor.inference_on_dataset(val_dataset, config.results_dir)
    
    # Print final results
    print("\nFinal Results:")
    print(f"DSC: {results_df['dsc'].mean():.3f} ± {results_df['dsc'].std():.3f}")
    print(f"MSE: {results_df['mse'].mean():.3e} ± {results_df['mse'].std():.3e}")
    print(f"NCC: {results_df['ncc'].mean():.3f} ± {results_df['ncc'].std():.3f}")
    print(f"TRE: {results_df['mean_tre'].mean():.3f} ± {results_df['mean_tre'].std():.3f} mm")
    print(f"Max TRE: {results_df['max_tre'].mean():.3f} ± {results_df['max_tre'].std():.3f} mm")
    
    # Plot training curves
    print("\nGenerating training curves...")
    visualizer.plot_training_curves(metrics_history)

    print("\nGenerating visualizations...")
    for idx in range(len(val_dataset)):
        data = val_dataset[idx]
        prediction = predictor.predict_case(data)
        visualizer.visualize_case(prediction, idx + config.test_start_idx)
    
    print(f"\nResults saved in: {config.results_dir}")

if __name__ == "__main__":
    main()