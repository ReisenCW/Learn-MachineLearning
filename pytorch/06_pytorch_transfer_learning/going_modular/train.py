import os
import torch
from torch import nn
from pathlib import Path
import data_setup, engine, model_builder, utils

import torchvision.transforms.v2 as T

MODEL_DIR = Path("models")
MODEL_NAME = "tiny_vgg.pt"

DATA_DIR = Path("../data/pizza_steak_sushi/")
TRAIN_PATH = DATA_DIR / "train"
TEST_PATH = DATA_DIR / "test"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

HIDDEN_UNITS = 8

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


