import cv2
import numpy as np
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple

from tqdm import tqdm


from skimage import exposure

from utils import downscale, empty, show_text, ColorDetector


def process_image(img, yellow: ColorDetector):
    cv2.imshow("Original", img)

    # convert to hsv for better colour analysis
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # extract all the colours that we need
    yellow_filtered = cv2.inRange(hsv_img, yellow.lower_bounds, yellow.upper_bounds)

    cv2.imshow("Yellow", yellow_filtered)


def main():

    image = cv2.imread("./images/all_zones_vert.jpg")
    image = downscale(image, 10)

    start_time = perf_counter()
    fps = 30

    cv2.namedWindow("Trackbars")
    yellow_detector = ColorDetector(
        "Yellow", "Trackbars", (22, 210, 170), (45, 255, 255)
    )

    while True:

        current_time = perf_counter()
        duration = current_time - start_time
        start_time = current_time

        new_fps = 1 / duration

        if abs(new_fps - fps) >= 2:
            fps = new_fps
            print(f"FPS: {fps}")

        yellow_detector.update_bounds()
        process_image(image, yellow_detector)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
