import cv2
import numpy as np
from time import perf_counter

from skimage import exposure

from utils import downscale, empty


def find_window(image, canny):
    max_area = 5000
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # cv2.circle(image, target_point, 10, (0, 0, 0), 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area >= max_area:
            perimeter = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
            corners = len(approx)

            x, y, width, height = cv2.boundingRect(approx)

            if corners == 4:
                aspect_ratio = width / float(height)
                aspect_ratio = np.round(aspect_ratio, 3)

                if aspect_ratio > 0.85 and aspect_ratio < 1.15:
                    # print(f"{aspect_ratio = }")
                    cv2.circle(
                        image,
                        (int(x) + int(width) // 2, int(y) + int(height) // 2),
                        10,
                        (0, 0, 0),
                        2,
                    )

                    cv2.rectangle(
                        image, (x, y), (x + width, y + height), (0, 255, 0), 3
                    )


def three_channels(image: cv2.Mat):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        return image.copy()
    else:
        raise NotImplementedError("WTF is an image with neither 2 nor 3 dimensions ?")


def process_image(image: cv2.Mat, win_name: str):
    result = image.copy()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = exposure.equalize_hist(grayscale)
    hist *= 255
    hist = np.asarray(hist, dtype=np.uint8)
    hist = cv2.erode(hist, (3, 3), iterations=1)

    threshold = cv2.getTrackbarPos("Threshold", "Trackbars")
    hist = (hist < threshold) * 255
    hist = np.asarray(hist, dtype=np.uint8)

    hist = cv2.erode(hist, (7, 7), iterations=2)

    # d = 5
    # hist = cv2.bilateralFilter(hist, d, 1000, 200)
    # hist = cv2.dilate(hist, (7, 7))

    canny_hist = cv2.Canny(hist, 250, 100)

    horizontal = np.copy(canny_hist)
    vertical = np.copy(canny_hist)

    filter_width = cv2.getTrackbarPos("Filter Width", "Trackbars")

    cols = horizontal.shape[1]
    h_size = cols // filter_width
    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))

    horizontal = cv2.erode(horizontal, h_structure)
    horizontal = cv2.dilate(horizontal, h_structure)

    lines = vertical.shape[0]
    v_size = lines // filter_width
    v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))

    vertical = cv2.erode(vertical, v_struct)
    vertical = cv2.dilate(vertical, v_struct)

    combined = cv2.bitwise_or(horizontal, vertical)
    combined = cv2.dilate(combined, (7, 7), iterations=3)

    """
    min_line_length = cv2.getTrackbarPos("Min line length", "Trackbars")
    max_line_gap = cv2.getTrackbarPos("Max line gap", "Trackbars")
    hough_thr = cv2.getTrackbarPos("Hough threshold", "Trackbars")

    
    hough_lines = cv2.HoughLines(
        combined, 1, np.pi / 180, 40, min_line_length, max_line_gap
    )
    #lines = cv2.HoughLinesP(
    #    combined, 1, np.pi / 180, hough_thr, min_line_length, max_line_gap
    #)

    try:
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:  # weird triple list
                    cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
    except TypeError:
        pass

    contours, hierarchy = cv2.findContours(
        combined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    min_area = cv2.getTrackbarPos("Area", "Trackbars")

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            # drawing contours is a very big performance hit
            cv2.drawContours(result, cnt, -1, (0, 0, 255), 2)
    """

    result = np.concatenate(
        (
            three_channels(result),
            three_channels(hist),
            three_channels(combined),
        ),
        axis=1,
    )

    cv2.imshow(win_name, result)


def main():

    selected = range(3, 7)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("Threshold", "Trackbars", 24, 255, empty)
    cv2.createTrackbar("Area", "Trackbars", 65, 500, empty)
    cv2.createTrackbar("Filter Width", "Trackbars", 30, 100, empty)

    images = []
    for s in selected:
        file = f"./images/window_{s}.jpg"
        image = cv2.imread(file)
        image = downscale(image, 10)

        images.append((image, file))

    start_time = perf_counter()
    fps = 30

    while True:

        current_time = perf_counter()
        duration = current_time - start_time
        start_time = current_time

        new_fps = 1 / duration

        if abs(new_fps - fps) >= 2:
            fps = new_fps
            print(f"FPS: {fps}")

        for (i, f) in images:
            process_image(i, f)

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
    stats.dump_stats("./win_detect.prof")
