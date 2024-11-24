# Lung CT Registration Project

This project implements a deep learning-based deformable image registration method for lung CT scans. The goal is to accurately align different CT images of the same patient taken at different times or breathing phases, which is crucial for various medical applications such as disease progression monitoring, treatment planning, and response assessment.

## Key Features

- Deformable 3D image registration
- Multi-resolution UNet-based architecture
- Multiple similarity metrics
- Keypoint-based validation
- Comprehensive visualization tools
- Detailed performance metrics

## Technical Details

### Registration Framework

The registration pipeline consists of several key components:

1. **Network Architecture**
   - 3D UNet for deformation field prediction
   - Multi-scale feature extraction
   - Skip connections for preserving spatial information
   - Output: dense 3D displacement field (DVF)

2. **Loss Functions**
   - Image similarity (Global Mutual Information)
   - Deformation smoothness (Bending Energy)
   - Segmentation consistency (Dice Loss)
   - Keypoint alignment (MSE)

3. **Evaluation Metrics**
   - Dice Similarity Coefficient (DSC)
   - Target Registration Error (TRE)
   - Mean Squared Error (MSE)
   - Normalized Cross Correlation (NCC)

4. **Data Processing**
   - Intensity normalization
   - Mask binarization
   - Keypoint correspondence tracking
   - Data augmentation

## Dataset

The dataset used in this project is from the Learn2Reg Challenge 2020, specifically : CT Lung Registration.

### Data Source
- Challenge Website: [Learn2Reg Challenge](https://learn2reg.grand-challenge.org/Datasets/)
- Task: CT Lung Registration (Inspiration-Expiration pairs)
- Dataset Size: 30 pairs of thoracic CT scans
  - 20 pairs for training
  - 10 pairs for testing

### Dataset Details
- 3D thoracic CT scan pairs in inspiration and expiration phases
- Resolution: ~1.75 × 1.75 × 2.5 mm
- Size: ~256 × 256 × 200 voxels
- Format: NIfTI (.nii.gz)


## Acknowledgments

- MONAI framework for medical imaging tools
- PyTorch for deep learning framework
