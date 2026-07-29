import numpy as np

PATCH_SIZE = 512
STRIDE = 256


def create_hanning_window(size: int = PATCH_SIZE):

    h = np.hanning(size)

    window = np.outer(h, h)

    window = window.astype(np.float32)

    return window


def extract_patches(image):

    patches = []

    positions = []

    height, width = image.shape[:2]

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):

        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            patch = image[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            patches.append(patch)

            positions.append((y, x))

    return patches, positions


def stitch_predictions(
    patch_predictions,
    patch_positions,
    image_shape,
    num_classes,
):
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

    for prediction, (y, x) in zip(
        patch_predictions,
        patch_positions,
    ):

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

    prediction_sum /= np.maximum(weight_sum, 1e-8)

    mask = np.argmax(
        prediction_sum,
        axis=0,
    )

    return mask
