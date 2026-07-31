import numpy as np

PATCH_SIZE = 512
STRIDE = 256


def create_hanning_window(size: int = PATCH_SIZE):
    h = np.hanning(size)
    window = np.outer(h, h)
    return window.astype(np.float32)


def extract_patches(image):
    """
    Generator that yields one patch at a time.

    Returns:
        patch, (y, x)
    """
    height, width = image.shape[:2]

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch = image[
                y : y + PATCH_SIZE,
                x : x + PATCH_SIZE,
            ]
            yield patch, (y, x)


def initialize_stitching(image_shape, num_classes):
    """
    Create the buffers used for stitching predictions.
    """
    height, width = image_shape[:2]

    prediction_sum = np.zeros(
        (num_classes, height, width),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    window = create_hanning_window()

    return prediction_sum, weight_sum, window


def add_patch_prediction(
    prediction_sum,
    weight_sum,
    window,
    prediction,
    position,
):
    """
    Add a single patch prediction to the final segmentation.
    """
    y, x = position

    prediction_sum[
        :,
        y : y + PATCH_SIZE,
        x : x + PATCH_SIZE,
    ] += (
        prediction * window
    )

    weight_sum[
        y : y + PATCH_SIZE,
        x : x + PATCH_SIZE,
    ] += window


def finalize_stitching(
    prediction_sum,
    weight_sum,
):
    """
    Convert accumulated probabilities into the final mask.
    """
    prediction_sum /= np.maximum(weight_sum, 1e-8)

    mask = np.argmax(
        prediction_sum,
        axis=0,
    )

    return mask
