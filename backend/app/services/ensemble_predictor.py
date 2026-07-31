import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import psutil

process = psutil.Process(os.getpid())


def log_memory(stage):
    print(f"{stage}: {process.memory_info().rss / 1024 / 1024:.1f} MB")


from app.utils.patching import (
    extract_patches,
    finalize_stitching,
    initialize_stitching,
    add_patch_prediction,
)
from app.utils.grading import compute_gleason_score
import base64
from io import BytesIO


class EnsemblePredictor:

    NUM_CLASSES = 6

    def __init__(
        self,
        onnx_path: str,
    ):

        so = ort.SessionOptions()

        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        print("✓ ONNX model loaded")

    def _preprocess(self, image):

        image = image.astype(np.float32)

        image /= 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32)

    def _predict_patch(self, patch):

        input_tensor = self._preprocess(patch)

        logits = self.session.run(
            None,
            {
                self.input_name: input_tensor,
            },
        )[0]

        # Softmax
        logits = logits - np.max(logits, axis=1, keepdims=True)

        exp = np.exp(logits)

        probs = exp / np.sum(exp, axis=1, keepdims=True)

        return probs.squeeze(0)

    def predict(self, image: Image.Image):

        log_memory("Start")

        image = image.convert("RGB")
        log_memory("After convert")

        image = image.resize((1024, 1024))
        log_memory("After resize")

        image_np = np.asarray(image)

        prediction_sum, weight_sum, window = initialize_stitching(
            image_np.shape,
            self.NUM_CLASSES,
        )

        for i, (patch, position) in enumerate(extract_patches(image_np)):

            prediction = self._predict_patch(patch)

            log_memory(f"After patch {i}")

            add_patch_prediction(
                prediction_sum,
                weight_sum,
                window,
                prediction,
                position,
            )

        mask = finalize_stitching(
            prediction_sum,
            weight_sum,
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
                [0, 128, 0],  # Benign
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
