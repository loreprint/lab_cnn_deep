from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class FaceDetectionResult:
    cropped_face: Image.Image
    boxed_image: Image.Image
    bbox: tuple[int, int, int, int]
    method: str


def detect_primary_face(
    image: Image.Image,
    margin_ratio: float = 0.18,
    min_face_size: int = 40,
    max_detection_size: int = 960,
) -> FaceDetectionResult | None:
    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image)

    scale = 1.0
    if max(rgb_image.width, rgb_image.height) > max_detection_size:
        scale = max_detection_size / max(rgb_image.width, rgb_image.height)
        resized_width = max(1, int(rgb_image.width * scale))
        resized_height = max(1, int(rgb_image.height * scale))
        image_array = cv2.resize(image_array, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)
    # Keep the crop focused on the face with only a moderate amount of
    # surrounding context. Too much torso/background made the model brittle on
    # external photos, while a slightly taller crop preserves hairline, jaw,
    # and a small neck region that still helps classification.
    crop_width = int(w * (1.55 + margin_ratio))
    crop_height = int(h * (1.95 + margin_ratio))
    center_x = x + (w / 2)
    center_y = y + (h * 0.56)

    left = max(0, int(center_x - (crop_width / 2)))
    top = max(0, int(center_y - (crop_height * 0.45)))
    right = min(image.width, left + crop_width)
    bottom = min(image.height, top + crop_height)

    left = max(0, right - crop_width)
    top = max(0, bottom - crop_height)

    cropped_face = rgb_image.crop((left, top, right, bottom))

    boxed_image = rgb_image.copy()
    draw = ImageDraw.Draw(boxed_image)
    draw.rectangle((left, top, right, bottom), outline=(196, 72, 72), width=6)

    return FaceDetectionResult(
        cropped_face=cropped_face,
        boxed_image=boxed_image,
        bbox=(left, top, right, bottom),
        method="face_detection",
    )


def make_portrait_focus_crop(
    image: Image.Image,
    width_ratio: float = 0.44,
    height_ratio: float = 0.60,
    vertical_center_ratio: float = 0.33,
) -> FaceDetectionResult:
    rgb_image = image.convert("RGB")
    crop_width = max(1, int(rgb_image.width * width_ratio))
    crop_height = max(1, int(rgb_image.height * height_ratio))

    center_x = rgb_image.width // 2
    center_y = int(rgb_image.height * vertical_center_ratio)

    left = max(0, center_x - crop_width // 2)
    top = max(0, center_y - crop_height // 2)
    right = min(rgb_image.width, left + crop_width)
    bottom = min(rgb_image.height, top + crop_height)

    left = max(0, right - crop_width)
    top = max(0, bottom - crop_height)

    cropped_face = rgb_image.crop((left, top, right, bottom))

    boxed_image = rgb_image.copy()
    draw = ImageDraw.Draw(boxed_image)
    draw.rectangle((left, top, right, bottom), outline=(228, 152, 65), width=6)

    return FaceDetectionResult(
        cropped_face=cropped_face,
        boxed_image=boxed_image,
        bbox=(left, top, right, bottom),
        method="portrait_fallback",
    )
