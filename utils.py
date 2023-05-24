from typing import Tuple
import cv2
import numpy as np
from numba import jit


def empty(_):
    pass


class ColorDetector:
    def __init__(
        self,
        name: str,
        window: str,
        lower_bounds=(0, 0, 0),
        upper_bounds=(180, 255, 255),
    ) -> None:

        self.name = name
        self.window = window
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

        cv2.createTrackbar(
            "Hue Min " + self.name, self.window, self.lower_bounds[0], 180, empty
        )
        cv2.createTrackbar(
            "Hue Max " + self.name, self.window, self.upper_bounds[0], 180, empty
        )
        cv2.createTrackbar(
            "Sat Min " + self.name, self.window, self.lower_bounds[1], 255, empty
        )
        cv2.createTrackbar(
            "Sat Max " + self.name, self.window, self.upper_bounds[1], 255, empty
        )
        cv2.createTrackbar(
            "Val Min " + self.name, self.window, self.lower_bounds[2], 255, empty
        )
        cv2.createTrackbar(
            "Val Max " + self.name, self.window, self.upper_bounds[2], 255, empty
        )

    def update_bounds(self):
        self.lower_bounds[0] = cv2.getTrackbarPos("Hue Min " + self.name, self.window)
        self.upper_bounds[0] = cv2.getTrackbarPos("Hue Max " + self.name, self.window)
        self.lower_bounds[1] = cv2.getTrackbarPos("Sat Min " + self.name, self.window)
        self.upper_bounds[1] = cv2.getTrackbarPos("Sat Max " + self.name, self.window)
        self.lower_bounds[2] = cv2.getTrackbarPos("Val Min " + self.name, self.window)
        self.upper_bounds[2] = cv2.getTrackbarPos("Val Max " + self.name, self.window)


def show_text(canvas, text: str, point: Tuple[int, int], color: Tuple[int, int, int]):
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1)


def downscale(image: cv2.Mat, factor: float, interpolation=cv2.INTER_LINEAR) -> cv2.Mat:

    (h, w, _) = image.shape
    new_size = (w // factor, h // factor)

    return cv2.resize(image, new_size, interpolation=interpolation)


def show_text_box(canvas, text: str, rect_ul: np.ndarray, color: Tuple[int, int, int]):
    upper_left = np.array(rect_ul - np.array([0, 20]), dtype=np.int32)
    down_right = np.array(rect_ul + np.array([15 * len(text), 0]), dtype=np.int32)

    cv2.rectangle(canvas, upper_left, down_right, color, thickness=cv2.FILLED)
    show_text(canvas, text, (int(rect_ul[0]), int(rect_ul[1] + 1)), (0, 0, 0))
