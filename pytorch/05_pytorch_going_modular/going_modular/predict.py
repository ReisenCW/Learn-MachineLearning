import argparse
import torch
from torch import nn
from pathlib import Path
from torchvision.io import read_image
import torchvision.transforms.v2 as T
import model_builder

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image",
                    type=str,
                    help="target image filepath to predict on")

# Get a model path
parser.add_argument("-m","--model_path",
                    default="models/tiny_vgg.pt",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

IMAGE_PATH = args.image
MODEL_PATH = args.model_path

print(f"Predicting {IMAGE_PATH} with model {MODEL_PATH}")

class_names = ["pizza", "steak", "sushi"]
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(filepath: str | Path=MODEL_PATH) -> nn.Module:
    model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=128,
                                output_shape=len(class_names)).to(device)

    model.load_state_dict(torch.load(filepath))
    return model

def predict_on_image(image_path=IMAGE_PATH, model_path=MODEL_PATH):
    transform = T.Compose([
        T.Resize(size=(64, 64)),
        T.ToImage(),
        T.ToDtype(torch.float32, True)
    ])
    image = transform(read_image(image_path))
    model = load_model(model_path)
    model.eval()
    with torch.inference_mode():
        y_logit = model(image.unsqueeze(dim=0).to(device))
        pred_prob = torch.softmax(y_logit, dim=1)
        pred_class = class_names[pred_prob.argmax(dim=1)]
        print(f"[INFO] Pred class: {pred_class}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
    predict_on_image()