"""
Evaluation module for U-Net model performance assessment.
Adapted from https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(model, dataloader, device, amp=False):
    """Evaluate the model on the provided dataloader.
    
    Args:
        model: The neural network model to evaluate
        dataloader: DataLoader containing validation/test data
        device: Device to perform computations on (cuda/cpu)
        amp: Whether to use automatic mixed precision
        
    Returns:
        float: Mean Dice score across the dataset
    """
    model.eval()
    num_batches = len(dataloader)
    dice_score = 0

    # Progress bar for validation
    progress_bar = tqdm(
        dataloader, 
        total=num_batches, 
        desc='Evaluation', 
        unit='batch', 
        leave=False
    )

    # Automatic mixed precision context
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in progress_bar:
            # Extract data from batch
            images, masks_true = batch[0], batch[1]

            # Transfer to device with optimal memory format
            images = images.to(
                device=device, 
                dtype=torch.float32, 
                memory_format=torch.channels_last
            )
            masks_true = masks_true.to(device=device, dtype=torch.long)

            # Generate predictions
            masks_pred = model(images)

            # Handle binary vs multiclass case
            if model.n_classes == 1:
                # Binary segmentation case
                assert masks_true.min() >= 0 and masks_true.max() <= 1, \
                    'Binary mask should have values in [0, 1]'
                
                # Apply sigmoid and threshold for binary prediction
                masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
                
                # Compute binary Dice score
                dice_score += dice_coeff(
                    masks_pred, 
                    masks_true, 
                    reduce_batch_first=False
                )
            else:
                # Multiclass segmentation case
                assert masks_true.min() >= 0 and masks_true.max() < model.n_classes, \
                    f'Class indices should be in [0, {model.n_classes-1}]'
                
                # Convert ground truth to one-hot format
                masks_true = F.one_hot(
                    torch.squeeze(masks_true), 
                    model.n_classes
                ).permute(0, 3, 1, 2).float()
                
                # Convert prediction to one-hot format
                masks_pred = F.one_hot(
                    masks_pred.argmax(dim=1), 
                    model.n_classes
                ).permute(0, 3, 1, 2).float()
                
                # Compute multiclass Dice score, ignoring background class
                dice_score += multiclass_dice_coeff(
                    masks_pred[:, 1:], 
                    masks_true[:, 1:], 
                    reduce_batch_first=False
                )

    # Reset model to training mode 
    model.train()
    
    # Return average Dice score
    return dice_score / max(num_batches, 1)
