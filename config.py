import torch.nn as nn

# Paths
H5_FILE_PATH = '/home/arawa/Segmentation_shabaz_FV/2D_unet_tiles/training_data.h5'
MODEL_SAVE_PATH = '/home/arawa/Segmentation_shabaz_FV/output/3Dunet_trained_model_FV3.pth'
LOG_DIR = './logs'

# Hyperparameters
IN_CHANNELS = 1
OUT_CHANNELS = 1
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
VOLUME_DEPTH = 4
BATCH_SIZE = 8
LOSS_FUNCTION = nn.BCEWithLogitsLoss()