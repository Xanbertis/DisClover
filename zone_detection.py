import cv2
import numpy as np
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple

from tqdm import tqdm


from skimage import exposure
from skimage.metrics import (
    normalized_mutual_information,
    structural_similarity,
    peak_signal_noise_ratio,
    normalized_root_mse,
)
from skimage.transform import rotate

from utils import downscale, empty, show_text, ColorDetector, show_text_box

yellow_ref = cv2.imread("refs/yellow.png")
yellow_ref = cv2.cvtColor(yellow_ref, cv2.COLOR_BGR2GRAY)
yellow_ref = exposure.equalize_adapthist(yellow_ref)
yellow_ref_hflip = rotate(yellow_ref, angle=180)

# cv2.imshow("ref", yellow_ref)
# cv2.imshow("flipped ref", yellow_ref_hflip)


def compute_similarity(ref, image):
    return 10 * (normalized_mutual_information(ref, image) - 1)


def find_yellow_target(hsv_img, yellow: ColorDetector, canvas):
    yellow_filtered = cv2.inRange(hsv_img, yellow.lower_bounds, yellow.upper_bounds)

    # cv2.imshow("Yellow", yellow_filtered)

    contours, _ = cv2.findContours(
        yellow_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, c in enumerate(contours):

        if cv2.contourArea(c) < 20:
            continue

        box = cv2.boxPoints(cv2.minAreaRect(c))
        ibox = np.intp(box)

        src_pts = np.array([box[0], box[1], box[2]], dtype=np.float32)
        dest_pts = np.array([[0, 0], [210, 0], [210, 297]], dtype=np.float32)

        transform = cv2.getAffineTransform(src_pts, dest_pts)

        target_zone = cv2.warpAffine(hsv_img, transform, (210, 297))
        target_zone_bgr = cv2.cvtColor(target_zone, cv2.COLOR_HSV2BGR)
        target_zone_bgr = cv2.cvtColor(target_zone_bgr, cv2.COLOR_BGR2GRAY)

        target_zone_bgr = exposure.equalize_adapthist(target_zone_bgr)
        target_zone_bgr = np.array(target_zone_bgr * 255, dtype=np.uint8)

        lscore = compute_similarity(yellow_ref, target_zone_bgr)
        rscore = compute_similarity(yellow_ref_hflip, target_zone_bgr)

        if lscore > rscore:
            score = lscore
        else:
            score = rscore

        """
        show_text(
            target_zone_bgr,
            f"{direction} {score:.3f} {other:.3f} 68",
            (20, 20),
            (0, 0, 0),
        )

        cv2.imshow(f"Target {i}", target_zone_bgr)
        """

        cv2.drawContours(canvas, [ibox], 0, (0, 255, 255), thickness=2)
        show_text_box(canvas, f"{score:.3f}", box[0], (0, 255, 255))


def process_image(img, yellow: ColorDetector):
    # cv2.imshow("Original", img)

    result = img.copy()

    # convert to hsv for better colour analysis
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # find targets
    find_yellow_target(hsv_img, yellow, result)

    # display results
    cv2.imshow("Result", result)


def main():

    image = cv2.imread("./images/all_fake_vert.jpg")
    image = downscale(image, 2)

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
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.dump_stats("./zone_detect.prof")
