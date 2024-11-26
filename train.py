
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# From your own files/code
from UNet3D import UNet3D
from dataset_augumented import LiverVolumeDataset  
from torch.utils.data import DataLoader, random_split
from utils import (visualize_predictions,
    dice_coefficient, iou_score, precision_score,
    recall_score, f1_score, initialize_weights, initialize_tensorboard, log_validation_metrics,
    log_model_parameters, log_learning_rate, close_tensorboard,
    calculate_and_accumulate_metrics, update_validation_progress_bar, compute_avg_validation_metrics, print_metrics_val, calculate_print_log_train_metrics
)
from config import (
    H5_FILE_PATH, MODEL_SAVE_PATH, LOG_DIR, IN_CHANNELS, OUT_CHANNELS,
    NUM_EPOCHS, VOLUME_DEPTH, BATCH_SIZE,  LEARNING_RATE, LOSS_FUNCTION
)

# Set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use GPU 1,2,3 or 4

# Clear GPU cache
torch.cuda.empty_cache()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)  # Instantiate model
model.apply(initialize_weights)
model = model.to(DEVICE)
OPTIMIZER = optim.AdamW(model.parameters(), LEARNING_RATE)
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, mode='min', factor=0.1, patience=10, verbose=True) 
dataset = LiverVolumeDataset(h5_file_path=H5_FILE_PATH, volume_depth=VOLUME_DEPTH)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

writer = initialize_tensorboard(LOG_DIR, model, train_loader, DEVICE)  # Initialize TensorBoard

# TRAINING 
for epoch in range(NUM_EPOCHS):
    print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}", flush=True)
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training", unit="batch") as pbar:
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer to device
            labels = labels.float()  # Convert labels to float for loss function
            OPTIMIZER.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            if labels.shape != outputs.shape:  # Resize the labels to match the output shape
                labels = F.interpolate(labels, size=outputs.shape[2:], mode='trilinear')
            loss = LOSS_FUNCTION(outputs, labels)  # Calculate the loss
            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            OPTIMIZER.step()  # Optimization
            running_loss += loss.item()  # Accumulate the loss

            # Calculate metrics
            dice = dice_coefficient(outputs, labels)
            iou = iou_score(outputs, labels)
            precision = precision_score(outputs, labels)
            recall = recall_score(outputs, labels)
            f1 = f1_score(outputs, labels)
            running_dice += dice
            running_iou += iou
            running_precision += precision
            running_recall += recall
            running_f1 += f1
            pbar.set_postfix(
                loss=running_loss / (i + 1),
                dice=running_dice / (i + 1),
                iou=running_iou / (i + 1),
                precision=running_precision / (i + 1),
                recall=running_recall / (i + 1),
                f1=running_f1 / (i + 1)
            )
            pbar.update(1)
    calculate_print_log_train_metrics(writer, epoch, NUM_EPOCHS, train_loader, running_loss, running_dice,running_iou, running_precision, running_recall, running_f1)  # Calculate and log training metrics

    # VALIDATION
    model.eval()
    val_loss = 0.0
    metrics_dict = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation", unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer to device
                labels = labels.float()  # Convert labels to float for loss function
                outputs = model(inputs)  # Forward pass
                if labels.shape != outputs.shape:  # Resize the labels to match the output shape
                    labels = F.interpolate(labels, size=outputs.shape[2:], mode='trilinear')
                loss = LOSS_FUNCTION(outputs, labels)  # Compute loss
                val_loss += loss.item()  # Accumulate the loss
                metrics_dict = calculate_and_accumulate_metrics(outputs, labels, metrics_dict)  # Calculate and accumulate metrics
                update_validation_progress_bar(pbar, i, val_loss, metrics_dict)  # Update progress bar
                visualize_predictions(inputs, labels, outputs, writer, epoch)  # Visualize predictions
                pbar.update(1)

    avg_val_loss, avg_metrics = compute_avg_validation_metrics(val_loss, metrics_dict, len(val_loader))  # Compute average validation metrics
    print_metrics_val(epoch, NUM_EPOCHS, avg_val_loss, avg_metrics) 

    log_validation_metrics(
        writer, epoch, avg_val_loss, avg_metrics['dice'], avg_metrics['iou'],
        avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1']
    )  # Log validation metrics to TensorBoard

    log_model_parameters(writer, model, epoch)  # Log histograms of model parameters and gradients
    log_learning_rate(writer, OPTIMIZER, epoch)  # Log learning rate
    SCHEDULER.step(avg_val_loss)  # Adjust learning rate
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{'/home/arawa/Segmentation_shabaz_FV/models_every10epochs/'}_epoch_{epoch}_FV3.pth")


# Save trained model
try:
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}", flush=True)
except Exception as e:
    print(f"Error saving model: {e}", flush=True)

# Close TensorBoard writer
close_tensorboard(writer)
print("Training complete!", flush=True)