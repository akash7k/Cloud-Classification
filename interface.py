
# Model
from model import DeepResNet as TheModel

# Training function
from train import train_model as the_trainer

# Prediction function
from predict import cryptic_inf_f as the_predictor

# Dataset class
from dataset import ApplyTransform as TheDataset

# Dataloader function
from dataset import unicornLoader as the_dataloader

# Config variables
from config import batchsize as the_batch_size
from config import epochs as total_epochs
