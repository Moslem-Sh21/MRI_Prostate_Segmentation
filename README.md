# Prostate MRI Segmentation Framework

A deep learning-based framework for prostate and lesion segmentation in multi-parametric MRI (mpMRI) using PyTorch and U-Net architecture.

## Overview

This framework provides tools for training and evaluating U-Net models on prostate MRI data using three different MRI sequences:
- T2-weighted (T2w) - for prostate segmentation
- Apparent Diffusion Coefficient (ADC) - for lesion segmentation
- High b-value (HBV) - for lesion segmentation

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/prostate-mri-segmentation.git
cd prostate-mri-segmentation

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- PyTorch 1.10.1+
- torchvision 0.11.2+
- NumPy 1.22.0+
- SimpleITK 2.2.1+
- matplotlib
- tqdm

## Dataset Structure

The framework is designed to work with the PI-CAI dataset. The expected directory structure is:

```
PICAI_dataset/
├── picai_labels-main/
│   ├── anatomical_delineations/
│   │   └── whole_gland/
│   │       └── AI/
│   │           └── Bosma22b/
│   ├── csPCa_lesion_delineations/
│   │   └── AI/
│   │       └── Bosma22a/
│   └── clinical_information/
├── picai_public_images_fold0/
├── picai_public_images_fold1/
├── picai_public_images_fold2/
├── picai_public_images_fold3/
└── picai_public_images_fold4/
```

## Usage

### Training

Train a model for T2w data (prostate segmentation):

```bash
python train_eval.py \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --epochs 50 \
    --data 'T2w' \
    --loss 'dice_loss' \
    --data_path '/path/to/PICAI_dataset/' \
    --momentum 0.9 \
    --weight_decay 2e-4 \
    --save_step 10 \
    --min_delta 0.07
```

Train a model for ADC data (lesion segmentation):

```bash
python train_eval.py \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --epochs 50 \
    --data 'Adc' \
    --loss 'dice_loss' \
    --data_path '/path/to/PICAI_dataset/' \
    --momentum 0.9 \
    --weight_decay 2e-4 \
    --save_step 10 \
    --min_delta 0.07
```

Train a model for HBV data (lesion segmentation):

```bash
python train_eval.py \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --epochs 50 \
    --data 'Hbv' \
    --loss 'dice_loss' \
    --data_path '/path/to/PICAI_dataset/' \
    --momentum 0.9 \
    --weight_decay 2e-4 \
    --save_step 10 \
    --min_delta 0.07
```

### Visualizing Results

After training, visualize the segmentation results:

```bash
python visualize_results.py --data_path '/path/to/PICAI_dataset/'
```

## Project Structure

- `train_eval.py`: Main script for training and evaluating models
- `evaluate.py`: Functions for model evaluation
- `unet.py`: U-Net model architecture implementation
- `visualize_results.py`: Script for visualizing segmentation results
- `utils/`: Utility functions for data processing and metrics calculation

## Implementation Details

- **Data Preprocessing**: MRI volumes are resampled to standardized spacing and cropped to consistent dimensions
- **Data Augmentation**: Basic augmentation techniques are applied to improve model generalization
- **Model Architecture**: U-Net with configurable parameters for encoder/decoder pathways
- **Loss Functions**: Dice loss for segmentation optimization
- **Evaluation Metrics**: Dice coefficient for quantitative assessment

## Results

The framework produces the following:
- Trained models saved in the `checkpoints/` directory
- Training and validation loss curves
- Segmentation visualizations for qualitative assessment
- Dice scores for quantitative evaluation

## Citation

If you use this code, please cite:
```
@article{your-reference,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={Year}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
