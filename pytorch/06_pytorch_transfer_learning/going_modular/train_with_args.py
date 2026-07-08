import os
import torch
from torch import nn
from pathlib import Path
import data_setup, engine, model_builder, utils
import argparse

import torchvision.transforms.v2 as T

# 参数解析器
parser = argparse.ArgumentParser(description="Training Hyperparameters")

parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help="number of samples per batch")

parser.add_argument("--num_epochs", 
                     default=10, 
                     type=int, 
                     help="the number of epochs to train for")

parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")

parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")

args = parser.parse_args()

MODEL_DIR = Path("models")
MODEL_NAME = "tiny_vgg.pt"

TRAIN_PATH = args.train_dir
TEST_PATH = args.test_dir

BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate

HIDDEN_UNITS = args.hidden_units

print(f"[INFO] Training a model for {EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")
print(f"[INFO] Training data file: {TRAIN_PATH}")
print(f"[INFO] Testing data file: {TEST_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize(size=(64, 64)),
    T.ToImage(),
    T.ToDtype(torch.float32, True)
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=TRAIN_PATH,
    test_dir=TEST_PATH,
    transform=transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir=MODEL_DIR,
                 model_name=MODEL_NAME)


