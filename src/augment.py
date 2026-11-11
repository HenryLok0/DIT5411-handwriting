"""OpenCV geometric augmentation: rotation, shear, and scaling.

Each training photo is expanded with a fixed list of transforms so that a
character with 40 real samples becomes 200 training samples (5 versions each).
Augmentation id 0 is the unchanged image. Further ids are deterministic, which
keeps the experiment repeatable.
"""

from __future__ import annotations

import numpy as np

# Five versions are enough for the common case: 40 real images x 5 = 200.
# Extra entries cover characters that have fewer than 40 usable training photos.
TRANSFORMS = [
    {"name": "original", "rot": 0.0, "shear_x": 0.0, "shear_y": 0.0, "scale": 1.0},
    {"name": "rotate_p12", "rot": 12.0, "shear_x": 0.0, "shear_y": 0.0, "scale": 1.0},
    {"name": "rotate_m12", "rot": -12.0, "shear_x": 0.0, "shear_y": 0.0, "scale": 1.0},
    {"name": "shear_x", "rot": 0.0, "shear_x": 0.25, "shear_y": 0.0, "scale": 1.0},
    {"name": "scale_080", "rot": 0.0, "shear_x": 0.0, "shear_y": 0.0, "scale": 0.80},
    {"name": "rotate_p20_shear", "rot": 20.0, "shear_x": 0.15, "shear_y": 0.0, "scale": 1.0},
    {"name": "rotate_m18_scale", "rot": -18.0, "shear_x": 0.0, "shear_y": 0.10, "scale": 1.10},
    {"name": "shear_y_scale", "rot": 0.0, "shear_x": -0.20, "shear_y": 0.0, "scale": 0.90},
    {"name": "rotate_p8_scale", "rot": 8.0, "shear_x": 0.10, "shear_y": -0.10, "scale": 1.15},
    {"name": "rotate_m25", "rot": -25.0, "shear_x": 0.0, "shear_y": 0.0, "scale": 0.85},
    {"name": "rotate_p25_shear", "rot": 25.0, "shear_x": 0.20, "shear_y": 0.0, "scale": 1.0},
    {"name": "shear_strong", "rot": 0.0, "shear_x": -0.30, "shear_y": 0.12, "scale": 0.75},
]


def params_for(aug_id: int) -> dict:
    """Return geometric parameters for one augmentation id."""
    if aug_id < 0:
        raise ValueError(f"aug_id must be >= 0, got {aug_id}")
    if aug_id < len(TRANSFORMS):
        return dict(TRANSFORMS[aug_id])

    # Classes with very few real photos need more than the base list.
    base = TRANSFORMS[aug_id % len(TRANSFORMS)]
    cycle = aug_id // len(TRANSFORMS)
    rot = float(base["rot"]) + cycle * 7.0
    shear_x = float(base["shear_x"]) * (1.0 if cycle % 2 == 0 else -1.0)
    shear_y = float(base["shear_y"]) + 0.05 * cycle
    scale = float(base["scale"]) + 0.03 * ((cycle % 5) - 2)
    return {
        "name": f"extra_{aug_id}",
        "rot": float(np.clip(rot, -35.0, 35.0)),
        "shear_x": float(np.clip(shear_x, -0.4, 0.4)),
        "shear_y": float(np.clip(shear_y, -0.4, 0.4)),
        "scale": float(np.clip(scale, 0.7, 1.3)),
    }


def background_value(image: np.ndarray) -> int:
    """Estimate the paper color from the four corners.

    Handwriting sheets are not always white, so a fixed border of 0 or 255
    paints a visible frame after rotation. The corner median follows the page.
    """
    height, width = image.shape[:2]
    span = max(1, min(5, height, width))
    corners = np.concatenate(
        [
            image[:span, :span].ravel(),
            image[:span, -span:].ravel(),
            image[-span:, :span].ravel(),
            image[-span:, -span:].ravel(),
        ]
    )
    return int(np.median(corners))


def transform_matrix(width: int, height: int, rot_deg: float, shear_x: float, shear_y: float, scale: float) -> np.ndarray:
    """Build the 2x3 matrix passed straight to cv2.warpAffine.

    This matches cv2.getRotationMatrix2D: scale below 1 shrinks the ink, and a
    positive angle rotates counter-clockwise. The composition order is scale,
    then shear, then rotation, about the image center.
    """
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    theta = np.deg2rad(rot_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scaling = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    shear = np.array(
        [[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Same linear map as cv2.getRotationMatrix2D.
    rotation = np.array(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    back = np.array(
        [[1.0, 0.0, center_x], [0.0, 1.0, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    forward = back @ rotation @ shear @ scaling @ to_origin
    return forward[:2].astype(np.float32)


def apply_transform(image: np.ndarray, aug_id: int) -> np.ndarray:
    """Apply one geometric transform. The array shape and dtype stay the same."""
    if image.ndim != 2:
        raise ValueError("apply_transform expects a grayscale image")
    if aug_id == 0:
        return image.copy()

    import cv2

    params = params_for(aug_id)
    height, width = image.shape[:2]
    matrix = transform_matrix(
        width,
        height,
        params["rot"],
        params["shear_x"],
        params["shear_y"],
        params["scale"],
    )
    fill = background_value(image)
    warped = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return warped


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a grayscale image to size x size."""
    import cv2

    if image.shape[0] == size and image.shape[1] == size:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def render_sample(image: np.ndarray, aug_id: int, img_size: int, work_size: int) -> np.ndarray:
    """Resize, augment at the working resolution, then resize to the model input.

    Geometric warps are done at ``work_size`` (default 128) so a 300x300 source
    is not thrown away before rotation, while the network still sees ``img_size``.
    """
    if image.ndim == 3:
        import cv2

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    working = resize_square(image, work_size)
    warped = apply_transform(working, aug_id)
    final = resize_square(warped, img_size)
    return np.ascontiguousarray(final, dtype=np.uint8)
