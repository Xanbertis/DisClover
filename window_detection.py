import cv2
import numpy as np
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple


from skimage import exposure

from utils import downscale, empty


@dataclass
class Rectangle:

    x: int
    y: int
    w: int
    h: int
    probability: float

    @classmethod
    def from_sides(cls, top, bottom, left, right):

        if left is None and right is None:
            # unable to create a rectangle without left or right
            return Rectangle(0, 0, 0, 0, 0.0)

        left_pos = left[0] + left[2] // 2
        right_pos = right[0] + right[2] // 2

        top_pos = None
        bottom_pos = None

        if top is not None:
            top_pos = top[1] + top[3] // 2
        else:
            if left[3] > right[3]:
                # take left as basis
                top_pos = left[1] - left[2] // 2
            else:
                # take right as a basis
                top_pos = right[1] - right[0] // 2

        if bottom is not None:
            bottom_pos = bottom[1] + bottom[3] // 2
        else:
            if left[3] > right[3]:
                # take left as a basis
                bottom_pos = left[1] + left[2] + left[3] // 2
            else:
                # take right as a basis
                bottom_pos = right[1] + right[2] + right[3] // 2

        return Rectangle(
            left_pos, top_pos, (right_pos - left_pos), (bottom_pos - top_pos), 1.0
        )

    def draw(self, canvas: cv2.Mat):
        cv2.rectangle(
            canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 0, 0), 2
        )


def _find_window(image, canny):
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


def random_junk():

    # d = 5
    # hist = cv2.bilateralFilter(hist, d, 1000, 200)
    # hist = cv2.dilate(hist, (7, 7))
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
    pass


def lines_filter(image: cv2.Mat, filter_width: int) -> cv2.Mat:
    canny_hist = cv2.Canny(image, 250, 100)

    horizontal = np.copy(canny_hist)
    vertical = np.copy(canny_hist)

    cols = horizontal.shape[1]
    h_size = cols // filter_width
    h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))

    horizontal = cv2.erode(horizontal, h_struct)
    horizontal = cv2.dilate(horizontal, h_struct)

    lines = vertical.shape[0]
    v_size = lines // filter_width
    v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))

    vertical = cv2.erode(vertical, v_struct)
    vertical = cv2.dilate(vertical, v_struct)

    combined = cv2.bitwise_or(horizontal, vertical)
    combined = cv2.dilate(combined, (7, 7), iterations=3)

    return combined


def sides_filter(image: cv2.Mat, filter_width: int) -> cv2.Mat:
    h_struct = np.array(
        [
            np.zeros(filter_width),
            np.ones(filter_width),
            np.zeros(filter_width),
        ],
        dtype=np.uint8,
    )

    v_struct = h_struct.copy().transpose()

    h_blocs = cv2.erode(image, h_struct)
    v_blocs = cv2.erode(image, v_struct)

    combined = cv2.bitwise_or(h_blocs, v_blocs)

    return combined


def remove_corners(image: cv2.Mat) -> cv2.Mat:
    corners = cv2.cornerHarris(image, 6, 3, 0.04)
    corners = cv2.dilate(corners, None)
    corners = np.where(corners > 0.01 * corners.max(), 0, 255)
    corners = np.asarray(corners, dtype=np.uint8)

    return cv2.bitwise_and(image, image, mask=corners)


def find_windows(image: cv2.Mat, canvas: cv2.Mat):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")

    rects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            rects.append(cv2.boundingRect(cnt))

    def rect_len(rect):
        _, _, w, h = rect
        if w > h:
            return w
        else:
            return h

    rects.sort(key=rect_len, reverse=True)

    def larger(a):
        return a > 1

    squares = []

    for current in rects:
        # iterate over all rects to find the best match for a square

        x0, y0, w0, h0 = current

        wh_ratio = w0 / h0

        if larger(wh_ratio):
            # we actually only care about vertical rects, since horizontal will anyway get included
            continue

        found_top = None
        found_bottom = None
        found_left = None
        found_right = None

        for r in rects:
            if current == r:
                continue

            x1, y1, w1, h1 = r
            wh1_ratio = w1 / h1

            if larger(wh1_ratio):
                if found_top is not None and found_bottom is not None:
                    continue

                v_pos = y1 + h1 // 2

                top_rect_mid = x0 - w0
                if (v_pos > (top_rect_mid - 1.5 * w0)) and (
                    v_pos < (top_rect_mid + 1.5 * w0)
                ):
                    found_top = r
                    continue

                bot_rect_mid = x0 + h0 + w0
                if (v_pos > (bot_rect_mid - 1.5 * w0)) and (
                    v_pos < (bot_rect_mid + 1.5 * w0)
                ):
                    found_bottom = r
                    continue

            else:
                if found_left is not None:
                    continue

                # horizontal space between rects
                h_spacing = (x1 + w1 // 2) - (x0 + w0 // 2)
                if abs(h_spacing) < 0.5 * h0 or abs(h_spacing) > 1.5 * h0:
                    # too close or too faar
                    continue

                # now we’re good
                if h_spacing > 0:
                    found_left = current
                    found_right = r
                else:
                    found_left = r
                    found_right = current

        # now that we found top, bottom, left, right, we need to build the square !
        new_square: Rectangle = Rectangle.from_sides(
            found_bottom, found_top, found_left, found_right
        )

        if new_square.probability > 0.01:
            squares.append(new_square)

    for s in squares:
        s.draw(canvas)


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

    filter_width = cv2.getTrackbarPos("Filter Width", "Trackbars")

    sides = sides_filter(hist, filter_width)

    without_corners = remove_corners(sides)
    without_corners = cv2.erode(without_corners, (11, 11))

    find_windows(without_corners, result)

    result = np.concatenate(
        (
            three_channels(result),
            # three_channels(hist),
            # three_channels(sides),
            # three_channels(without_corners),
        ),
        axis=1,
    )

    cv2.imshow(win_name, result)


def main():

    selected = range(3, 7)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("Threshold", "Trackbars", 24, 255, empty)
    cv2.createTrackbar("Min Area", "Trackbars", 65, 500, empty)
    cv2.createTrackbar("Filter Width", "Trackbars", 30, 100, empty)
    cv2.createTrackbar("Bar Width", "Trackbars", 20, 100, empty)

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
