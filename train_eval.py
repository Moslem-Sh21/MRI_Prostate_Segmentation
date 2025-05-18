#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Medical Image Segmentation Training and Evaluation Module

This module provides functionality for training and evaluating UNet models
for medical image segmentation tasks on MRI data.
"""

import os
import glob
import errno
import logging
import argparse
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from unet import UNet
from evaluate import evaluate
from utils.dice_score import dice_loss


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Stops training when validation loss exceeds training loss by more
    than min_delta for tolerance consecutive epochs.
    
    Args:
        tolerance: Number of consecutive epochs to wait before stopping
        min_delta: Minimum difference between validation and training loss
    """
    def __init__(self, tolerance: int = 5, min_delta: float = 0.0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss: float, validation_loss: float) -> None:
        """
        Check if training should stop.
        
        Args:
            train_loss: Current training loss
            validation_loss: Current validation loss
        """
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0


def mkdir_if_missing(dir_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path: Path to directory
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_folds(image_paths: List[str], mask_folder: List[str]) -> List[str]:
    """
    Match image paths with corresponding mask paths based on common identifiers.
    
    Args:
        image_paths: List of image file paths
        mask_folder: Folder containing mask files (first element used)
        
    Returns:
        List of matched mask file paths
    """
    matched_mask_paths = []
    
    for image_path in image_paths:
        # Get the base name of the file
        file_base = os.path.basename(image_path)
        
        # Split the base name into name and extension
        file_name, _ = os.path.splitext(file_base)
        
        # Extract identifiers from the filename
        parts = file_name.split("_")
        if len(parts) >= 2:
            part1, part2 = parts[0], parts[1]
            mask_path = f"{mask_folder[0]}{part1}_{part2}.nii"
            matched_mask_paths.append(mask_path)
        else:
            logger.warning(f"Filename format not as expected: {file_name}")
    
    return matched_mask_paths


def volume_resample_crop(
    image: sitk.Image, 
    spacing: List[float], 
    crop_size: List[int], 
    image_type: str
) -> sitk.Image:
    """
    Resample and crop a 3D medical image volume.
    
    Args:
        image: SimpleITK image to process
        spacing: Target spacing in mm (x, y, z)
        crop_size: Target image size in voxels (x, y, z)
        image_type: Type of image ("Lesion", "Prostate", "T2w", "Adc", or "Hbv")
        
    Returns:
        Resampled and cropped image
    """
    # Calculate new size based on original size and spacing ratio
    orig_size = np.array(image.GetSize())
    orig_spacing = image.GetSpacing()
    new_spacing = np.array(spacing)
    new_size = np.floor(orig_size * (orig_spacing / new_spacing)).astype(int)
    
    # Configure the resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(new_spacing))
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    
    # Use nearest neighbor for masks, B-spline for images
    if image_type in ["Lesion", "Prostate"]:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    
    # Execute resampling
    resampled_image = resampler.Execute(image)
    
    # Crop to target size around center
    size = resampled_image.GetSize()
    center = [int(size[i] / 2) for i in range(3)]
    start_index = [center[i] - int(crop_size[i] / 2) for i in range(3)]
    
    # Ensure valid start indices
    start_index = [max(0, idx) for idx in start_index]
    
    # Set cropping boundaries
    cropper = sitk.CropImageFilter()
    cropper.SetLowerBoundaryCropSize(start_index)
    cropper.SetUpperBoundaryCropSize([max(0, size[i] - (start_index[i] + crop_size[i])) for i in range(3)])
    
    # Crop the image
    cropped_image = cropper.Execute(resampled_image)
    
    return cropped_image


def slice_data(
    images: List[sitk.Image], 
    image_type: str, 
    fold_number: int, 
    spacing: List[float], 
    crop_size: List[int]
) -> List[sitk.Image]:
    """
    Resample, crop, and slice 3D volumes into 2D slices.
    
    Args:
        images: List of SimpleITK images to process
        image_type: Type of image ("Lesion", "Prostate", "T2w", "Adc", or "Hbv")
        fold_number: Fold identifier (0-4)
        spacing: Target spacing in mm (x, y, z)
        crop_size: Target image size in voxels (x, y, z)
        
    Returns:
        List of 2D slices extracted from the processed volumes
    """
    slices = []
    
    for volume in images:
        # Resample and crop the volume
        processed_volume = volume_resample_crop(volume, spacing, crop_size, image_type)
        
        # Extract slices along the z-axis
        for z in range(processed_volume.GetSize()[2]):
            z_slice = sitk.Extract(
                processed_volume, 
                [processed_volume.GetSize()[0], processed_volume.GetSize()[1], 0],
                [0, 0, z]
            )
            slices.append(z_slice)
    
    return slices


def compute_mean_and_std(images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and standard deviation of a dataset.
    
    Args:
        images: List of numpy arrays representing images
        
    Returns:
        Tuple of (mean, std) arrays for normalization
    """
    # Initialize arrays to accumulate mean and std
    mean_sum = None
    std_sum = None
    
    for i, image in enumerate(tqdm(images, desc="Computing statistics")):
        pixel_array = np.array(image)
        
        if i == 0:
            mean_sum = pixel_array.mean(axis=(0, 1))
            std_sum = pixel_array.std(axis=(0, 1))
        else:
            mean_sum += pixel_array.mean(axis=(0, 1))
            std_sum += pixel_array.std(axis=(0, 1))
    
    # Calculate mean values
    mean = mean_sum / len(images)
    std = std_sum / len(images)
    
    return mean, std


def augment_data(
    img_list: List[np.ndarray], 
    data_mean: np.ndarray, 
    data_std: np.ndarray, 
    data_type: str,
    use_augmentation: bool = True
) -> List[torch.Tensor]:
    """
    Apply data augmentation and normalization to images.
    
    Args:
        img_list: List of images as numpy arrays
        data_mean: Mean values for normalization
        data_std: Standard deviation values for normalization
        data_type: Type of data ("data" or "target")
        use_augmentation: Whether to apply augmentation transformations
        
    Returns:
        List of transformed tensor images
    """
    transforms = []
    
    # Add augmentation transforms if enabled
    if use_augmentation:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    
    # Add normalization for input data
    if data_type == "data":
        transforms.append(T.Normalize(data_mean, data_std))
    
    # Create transform composition
    transform = T.Compose(transforms)
    
    # Apply transformations
    transformed_imgs = []
    for img in img_list:
        float_tensor = T.ToTensor()(Image.fromarray(img)).float()
        img_tensor = transform(float_tensor)
        transformed_imgs.append(img_tensor)
    
    return transformed_imgs


def training_loop(
    args: argparse.Namespace, 
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """
    Train and validate the model.
    
    Args:
        args: Command line arguments
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (CPU or GPU)
        
    Returns:
        Lists of training and validation losses for each epoch
    """
    # Initialize metrics tracking
    train_loss_history = []
    validation_loss_history = []
    
    # Create log directory
    log_directory = os.path.join(args.log_dir, args.data)
    mkdir_if_missing(log_directory)
    
    # Configure optimizer and learning rate scheduler
    optimizer = optim.RMSprop(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay, 
        momentum=args.momentum
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2
    )
    
    # Set up mixed precision training if enabled
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Initialize tracking variables
    global_step = 0
    best_train_score = 0
    best_val_score = 0
    
    # Initialize early stopping if enabled
    early_stopping = EarlyStopping(
        tolerance=5, 
        min_delta=args.min_delta
    ) if args.early_stopping else None
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        # Training progress bar
        with tqdm(total=len(train_loader.dataset), 
                  desc=f'Epoch {epoch + 1}/{args.epochs}', 
                  unit='img') as pbar:
            
            # Process batches
            for batch in train_loader:
                images, true_masks = batch
                
                # Validate input shape
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels.'
                
                # Move data to device
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                # Forward pass with mixed precision if enabled
                with torch.cuda.amp.autocast(enabled=args.amp):
                    masks_pred = model(images)
                    loss = dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(
                            torch.squeeze(true_masks), 
                            model.n_classes
                        ).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
                
                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                # Update progress
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        
        # Calculate average epoch loss
        epoch_loss = epoch_loss / len(train_loader)
        
        # Evaluate performance
        val_score = evaluate(model, val_loader, device, args.amp)
        train_score = evaluate(model, train_loader, device, args.amp)
        
        # Update learning rate
        scheduler.step(val_score)
        
        # Track losses
        validation_loss = 1.0 - val_score.cpu().numpy()
        train_loss = 1.0 - train_score.cpu().numpy()
        validation_loss_history.append(validation_loss)
        train_loss_history.append(train_loss)
        
        # Log progress
        logger.info(f'Epoch {epoch+1}/{args.epochs}, '
                   f'Train loss: {train_loss:.4f}, '
                   f'Validation loss: {validation_loss:.4f}, '
                   f'LR: {optimizer.param_groups[0]["lr"]:.1e}')
        
        # Save checkpoint periodically
        if epoch % args.save_step == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(
                log_directory, 
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if train_score >= best_train_score and val_score >= best_val_score:
            best_train_score = train_score
            best_val_score = val_score
            best_model_path = os.path.join(log_directory, 'best.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with train score: {best_train_score:.4f}, "
                       f"val score: {best_val_score:.4f}")
        
        # Check for early stopping
        if early_stopping:
            early_stopping(train_loss, validation_loss)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Final evaluation
    logger.info("Training completed. Starting final evaluation...")
    val_score = evaluate(model, val_loader, device, args.amp)
    train_score = evaluate(model, train_loader, device, args.amp)
    
    # Log final scores
    logger.info(f'Final training Dice score: {train_score:.4f}')
    logger.info(f'Final validation Dice score: {val_score:.4f}')
    
    return train_loss_history, validation_loss_history


def plot_losses(train_losses: List[float], val_losses: List[float], save_path: str = None) -> None:
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Loss plot saved to {save_path}")
    else:
        plt.show()


def load_and_process_data(
    args: argparse.Namespace, 
    data_type: str, 
    output_spacing: List[float], 
    output_size: List[int]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and process data for the specified modality.
    
    Args:
        args: Command line arguments
        data_type: Type of image data ("T2w", "Adc", or "Hbv")
        output_spacing: Target spacing in mm (x, y, z)
        output_size: Target image size in voxels (x, y, z)
        
    Returns:
        DataLoaders for training, validation, and testing
    """
    # Data statistics (precomputed)
    data_stats = {
        "T2w": {"mean": 209.71, "std": 134.86, "mask_type": "Prostate"},
        "Adc": {"mean": 777.54, "std": 701.82, "mask_type": "Lesion"},
        "Hbv": {"mean": 11.81, "std": 7.65, "mask_type": "Lesion"},
    }
    
    if data_type not in data_stats:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    logger.info(f"Processing {data_type} data...")
    
    # File paths for images
    image_paths = {}
    for fold in range(5):
        pattern = os.path.join(
            args.data_path, 
            f"picai_public_images_fold{fold}/*/*{data_type.lower()}.mha"
        )
        image_paths[fold] = glob.glob(pattern)
        logger.info(f"Fold {fold}: Found {len(image_paths[fold])} {data_type} images")
    
    # Determine mask type and folder
    mask_type = data_stats[data_type]["mask_type"]
    if mask_type == "Prostate":
        mask_folder = glob.glob(os.path.join(
            args.data_path,
            "picai_labels-main/anatomical_delineations/whole_gland/AI/Bosma22b/"
        ))
    else:  # "Lesion"
        mask_folder = glob.glob(os.path.join(
            args.data_path,
            "picai_labels-main/csPCa_lesion_delineations/AI/Bosma22a/"
        ))
    
    # Match image and mask paths
    mask_paths = {}
    for fold in range(5):
        mask_paths[fold] = read_folds(image_paths[fold], mask_folder)
    
    # Load images and masks
    images = {}
    masks = {}
    for fold in range(5):
        logger.info(f"Loading fold {fold}...")
        images[fold] = [sitk.ReadImage(path) for path in tqdm(
            image_paths[fold], desc=f"Loading {data_type} images"
        )]
        masks[fold] = [sitk.ReadImage(path) for path in tqdm(
            mask_paths[fold], desc=f"Loading {mask_type} masks"
        )]
    
    # Process images and masks
    processed_images = {}
    processed_masks = {}
    for fold in range(5):
        logger.info(f"Processing fold {fold}...")
        processed_images[fold] = slice_data(
            images[fold], data_type, fold, output_spacing, output_size
        )
        del images[fold]  # Free memory
        
        processed_masks[fold] = slice_data(
            masks[fold], mask_type, fold, output_spacing, output_size
        )
        del masks[fold]  # Free memory
    
    # Convert to numpy arrays
    image_arrays = {}
    mask_arrays = {}
    for fold in range(5):
        logger.info(f"Converting fold {fold} to numpy arrays...")
        image_arrays[fold] = [sitk.GetArrayFromImage(img) for img in tqdm(
            processed_images[fold], desc=f"Converting {data_type} images"
        )]
        del processed_images[fold]  # Free memory
        
        mask_arrays[fold] = [sitk.GetArrayFromImage(mask) for mask in tqdm(
            processed_masks[fold], desc=f"Converting {mask_type} masks"
        )]
        del processed_masks[fold]  # Free memory
    
    # Define dataset splits (fold 1, 2, 4 for training, 3 for validation, 0 for testing)
    train_images = image_arrays[1] + image_arrays[2] + image_arrays[4]
    train_masks = mask_arrays[1] + mask_arrays[2] + mask_arrays[4]
    
    val_images = image_arrays[3]
    val_masks = mask_arrays[3]
    
    test_images = image_arrays[0]
    test_masks = mask_arrays[0]
    
    # Use precomputed statistics
    mean_value = data_stats[data_type]["mean"]
    std_value = data_stats[data_type]["std"]
    
    logger.info(f"Using statistics for {data_type}: mean={mean_value}, std={std_value}")
    
    # Apply transformations
    logger.info("Applying transformations...")
    transformed_train_images = augment_data(
        train_images, mean_value, std_value, "data", use_augmentation=True
    )
    del train_images  # Free memory
    
    transformed_val_images = augment_data(
        val_images, mean_value, std_value, "data", use_augmentation=False
    )
    del val_images  # Free memory
    
    transformed_test_images = augment_data(
        test_images, mean_value, std_value, "data", use_augmentation=False
    )
    del test_images  # Free memory
    
    transformed_train_masks = augment_data(
        train_masks, mean_value, std_value, "target", use_augmentation=True
    )
    del train_masks  # Free memory
    
    transformed_val_masks = augment_data(
        val_masks, mean_value, std_value, "target", use_augmentation=False
    )
    del val_masks  # Free memory
    
    transformed_test_masks = augment_data(
        test_masks, mean_value, std_value, "target", use_augmentation=False
    )
    del test_masks  # Free memory
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TensorDataset(
        torch.stack(transformed_train_images),
        torch.stack(transformed_train_masks)
    )
    
    val_dataset = TensorDataset(
        torch.stack(transformed_val_images),
        torch.stack(transformed_val_masks)
    )
    
    test_dataset = TensorDataset(
        torch.stack(transformed_test_images),
        torch.stack(transformed_test_masks)
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def main():
    """
    Main entry point for the training script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Medical Image Segmentation Training & Evaluation'
    )
    
    # Data parameters
    parser.add_argument('--data', type=str, default='T2w',
                        choices=['T2w', 'Adc', 'Hbv'],
                        help='Input data modality')
    parser.add_argument('--data_path', type=str, 
                        default='D:/Courses/CISC_881_Medical_Imaging/PICAI_dataset/',
                        help='Path to dataset')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--save_step', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--min_delta', type=float, default=0.07,
                        help='Minimum improvement for early stopping')
    
    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling in U-Net')
    
    # System parameters
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Pin memory for faster GPU transfer')
    parser.add_argument('--log_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set output spacing and size (in mm and voxels)
    output_spacing = [0.5, 0.5, 3.0]  # mm
    output_size = [300, 300, 16]      # voxels
    
    # Load and process data
    logger.info(f"Loading {args.data} dataset...")
    train_loader, val_loader, test_loader = load_and_process_data(
        args, args.data, output_spacing, output_size
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)
    logger.info(f"Model initialized: {model.__class__.__name__}")
    
    # Move model to device
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    logger.info("Starting training...")
    train_losses, val_losses = training_loop(
        args, model, train_loader, val_loader, device
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_score = evaluate(model, test_loader, device, args.amp)
    logger.info(f"Test Dice score: {test_score:.4f}")
    
    # Plot and save losses
    plot_losses(
        train_losses, 
        val_losses, 
        save_path=os.path.join(args.log_dir, args.data, 'loss_plot.png')
    )
    
    logger.info("Training and evaluation completed.")


if __name__ == '__main__':
    main()
