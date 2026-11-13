import torch


def predict(model, x):
    with torch.inference_mode():
        return model(x)
