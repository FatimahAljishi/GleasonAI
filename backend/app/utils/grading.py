import numpy as np


def compute_gleason_score(mask, min_fraction=0.006):
    total_pixels = mask.size
    values, counts = np.unique(mask, return_counts=True)
    label_counts = dict(zip(values, counts))

    min_pixels = total_pixels * min_fraction
    valid = {
        k: v for k, v in label_counts.items() if k not in [0, 2] and v >= min_pixels
    }

    if not valid:
        return "Unknown"

    if all(k == 1 for k in valid.keys()):
        return "Benign"

    valid = {k: v for k, v in valid.items() if k != 1}
    if not valid:
        return "Benign"

    top_two = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:2]
    grades = [label for label, _ in top_two]
    if len(grades) == 1:
        grades.append(grades[0])
    return f"{grades[0]}+{grades[1]}={sum(grades)}"
