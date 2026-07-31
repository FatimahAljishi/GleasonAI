from pathlib import Path
from xml.parsers.expat import model
import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from app.utils.patching import extract_patches, stitch_predictions
from app.utils.grading import compute_gleason_score
import base64
from io import BytesIO


class EnsemblePredictor:

    NUM_CLASSES = 6

    def __init__(
        self,
        efficientnet_checkpoint: str,
        device: str | None = None,
    ):

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.efficientnet = self._load_model(
            encoder_name="efficientnet-b4",
            checkpoint_path=efficientnet_checkpoint,
        )

        self.preprocess_eff = smp.encoders.get_preprocessing_fn(
            "efficientnet-b4",
            pretrained="imagenet",
        )

        print("✓ Model loaded")

    def _load_model(
        self,
        encoder_name: str,
        checkpoint_path: str,
    ):

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=self.NUM_CLASSES,
            activation=None,
        )

        weights = torch.load(checkpoint_path, map_location=self.device)

        model.load_state_dict(weights)

        model.to(self.device)
        model.eval()

        return model

    def _preprocess(self, image: Image.Image, preprocessing_fn):

        image = image.convert("RGB")

        image = np.asarray(image).astype(np.float32)

        image = preprocessing_fn(image)

        image = TF.to_tensor(image).float()

        image = image.unsqueeze(0)

        image = image.to(self.device)

        return image

    def _predict_patch(self, patch):

        # EfficientNet preprocessing
        eff_tensor = self._preprocess(
            patch,
            self.preprocess_eff,
        )

        with torch.no_grad():

            eff_logits = self.efficientnet(eff_tensor)

            eff_probs = F.softmax(eff_logits, dim=1)

        return eff_probs.squeeze(0).cpu().numpy()

    def predict(
        self,
        image: Image.Image,
    ):
        image = image.convert("RGB")

        image = image.resize((1024, 1024))

        image_np = np.asarray(image)

        patches, positions = extract_patches(image_np)

        predictions = []

        for patch in patches:

            patch_image = Image.fromarray(patch.astype(np.uint8))

            prediction = self._predict_patch(patch_image)

            predictions.append(prediction)

        mask = stitch_predictions(
            patch_predictions=predictions,
            patch_positions=positions,
            image_shape=image_np.shape,
            num_classes=self.NUM_CLASSES,
        )

        gleason = compute_gleason_score(mask)

        return {
            "mask": mask,
            "gleason_score": gleason,
        }

    def colorize_mask(self, mask):

        colors = np.array(
            [
                [255, 255, 255],  # Background
                [0, 255, 0],  # Benign
                [255, 165, 0],  # Unknown
                [0, 0, 255],  # Grade 3
                [255, 255, 0],  # Grade 4
                [255, 0, 0],  # Grade 5
            ],
            dtype=np.uint8,
        )

        return colors[mask]

    def _mask_to_base64(self, mask):

        colored = self.colorize_mask(mask)

        image = Image.fromarray(colored)

        buffer = BytesIO()

        image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return encoded
