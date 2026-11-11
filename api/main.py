# api/main.py
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model_definition import build_model


# Find the weights relative to this file, regardless of the terminal directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "mnist_model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the API starts.
    model = build_model()
    weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    app.state.model = model
    yield


app = FastAPI(title="Handwritten Digit API", lifespan=lifespan)


class Drawing(BaseModel):
    pixels: list[list[float]]


@app.post("/predict")
def predict(drawing: Drawing):
    if len(drawing.pixels) != 28 or any(len(row) != 28 for row in drawing.pixels):
        raise HTTPException(status_code=422, detail="Expected 28 rows of 28 pixels.")

    image = torch.tensor(drawing.pixels, dtype=torch.float32)

    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise HTTPException(status_code=422, detail="Pixels must be between 0 and 1.")

    if image.max() == 0:
        raise HTTPException(status_code=422, detail="Draw a digit before classifying.")

    # Match the normalization used when training on MNIST.
    image = (image - 0.1307) / 0.3081

    # Add batch and channel dimensions before flattening in the model.
    input_tensor = image.reshape(1, 1, 28, 28)

    with torch.inference_mode():
        scores = app.state.model(input_tensor)

    if scores.shape != (1, 10) or not torch.isfinite(scores).all():
        raise HTTPException(status_code=500, detail="Expected ten finite digit scores.")

    return {"digit": scores.argmax(dim=1).item()}