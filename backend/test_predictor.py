from app.services.ensemble_predictor import EnsemblePredictor
from PIL import Image
import numpy as np

predictor = EnsemblePredictor(
    efficientnet_checkpoint="model/unet_epoch8.pt",
    resnext_checkpoint="model/unet_epoch11.pt",
)

image = Image.open("test.jpg")

result = predictor.predict(image)

print(result["mask"].shape)

print(np.unique(result["mask"]))

print(result["gleason_score"])

colored = predictor.colorize_mask(result["mask"])

Image.fromarray(colored).save("prediction.png")
