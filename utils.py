from typing import Tuple
import cv2
import numpy as np


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

        def empty():
            pass

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
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
