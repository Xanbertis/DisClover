import cv2
import numpy as np
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple

from tqdm import tqdm


from skimage import exposure

from utils import downscale, empty, show_text


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


def sides_filter(image: cv2.Mat, filter_width: int) -> Tuple[cv2.Mat, cv2.Mat]:
    h_struct = np.array(
        [
            np.zeros(filter_width),
            np.ones(filter_width),
            np.zeros(filter_width),
        ],
        dtype=np.uint8,
    )

    v_struct = h_struct.T

    h_blocs = cv2.erode(image, h_struct)
    v_blocs = cv2.erode(image, v_struct)

    return h_blocs, v_blocs


def extract_rects(image: cv2.Mat, canvas: cv2.Mat):

    width, height = image.shape

    def find_rect(x, y, maxx, maxy):
        width = 0
        height = 1
        while x + width < maxx and image[x + width, y] == 0:
            width += 1

        while y + height < maxy:
            for w in range(x, x + width):
                if image[w, y + height] != 0:
                    break
            if image[x, y + height] != 0:
                break
            height += 1

        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 0, 255))

    for y in tqdm(range(16, height - 15)):
        for x in range(16, width - 15):
            if image[x, y] == 0:
                find_rect(x, y, width, height)


def remove_corners(image: cv2.Mat) -> cv2.Mat:
    corners = cv2.cornerHarris(image, 6, 3, 0.04)
    corners = cv2.dilate(corners, None)
    corners = np.where(corners > 0.01 * corners.max(), 0, 255)
    corners = np.asarray(corners, dtype=np.uint8)

    return cv2.bitwise_and(image, image, mask=corners)


def three_channels(image: cv2.Mat):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        return image.copy()
    else:
        raise NotImplementedError("WTF is an image with neither 2 nor 3 dimensions ?")


# note for later : use functional calls to reduce both function to only one that accepts
# a list of squares, and compute IOU of an arbitrary number of squares (take inspiration from the 4-way formula)


def bb_intersection_over_union(s1, s2, s3, s4):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(s1[0], s2[0], s3[0], s4[0])
    yA = max(s1[1], s2[1], s3[1], s4[1])
    xB = min(s1[2], s2[2], s3[2], s4[2])
    yB = min(s1[3], s2[3], s3[3], s4[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(s1[4] + s2[4] + s3[4] + s4[4] - 3 * interArea)
    # return the intersection over union value
    return iou


def iou_2(s1, s2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(s1[0], s2[0])
    yA = max(s1[1], s2[1])
    xB = min(s1[2], s2[2])
    yB = min(s1[3], s2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    area1 = (s1[2] - s1[0]) * (s1[3] - s1[1])
    area2 = (s2[2] - s2[0]) * (s2[3] - s2[1])

    iou = interArea / float(area1 + area2 - interArea)
    # return the intersection over union value
    return iou


def draw_rect(canvas, rect, color, **kwarg):
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, **kwarg)


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

    h_sides, v_sides = sides_filter(hist, filter_width)

    # reprocess vertical edges
    v_sides = remove_corners(v_sides)
    v_sides = cv2.erode(v_sides, (5, 5))
    v_sides = cv2.dilate(v_sides, (3, 3))

    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    contours, _ = cv2.findContours(v_sides, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    v_squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            # potentially suitable rect
            x, y, w, h = cv2.boundingRect(cnt)

            h_over_w = h / w
            # our target ratio is 3 (30 * 10 rectangle)
            # a ratio less than 1 is not acceptable
            # similarly is a ratio more than 10
            if h_over_w < 1 or h_over_w > 10:
                continue

            s1 = ((x + w, y), (x + w + h, y + h))
            s2 = ((x - h, y), (x, y + h))

            # cv2.rectangle(result, s1[0], s1[1], (255, 213, 115))
            # cv2.rectangle(result, s2[0], s2[1], (255, 213, 115))

            # cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0))

            rect_area = h * h

            v_squares.append((s1[0][0], s1[0][1], s1[1][0], s1[1][1], rect_area))
            v_squares.append((s2[0][0], s2[0][1], s2[1][0], s2[1][1], rect_area))

    contours, _ = cv2.findContours(h_sides, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    h_squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            # potentially suitable rect
            x, y, w, h = cv2.boundingRect(cnt)

            s1 = ((x, y + h), (x + w, y + h + w))
            s2 = ((x, y - w), (x + w, y))

            # cv2.rectangle(result, s1[0], s1[1], (51, 194, 255))
            # cv2.rectangle(result, s2[0], s2[1], (51, 194, 255))

            # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255))

            rect_area = w * w

            h_squares.append((s1[0][0], s1[0][1], s1[1][0], s1[1][1], rect_area))
            h_squares.append((s2[0][0], s2[0][1], s2[1][0], s2[1][1], rect_area))

    v_iou_matrix = np.zeros(
        (len(v_squares), len(v_squares), len(h_squares), len(h_squares))
    )

    # yup, this is n⁴ (actually n²m² but no one cares)
    for i, s1 in enumerate(v_squares):
        for j, s2 in enumerate(v_squares):
            for k, s3 in enumerate(h_squares):
                for l, s4 in enumerate(h_squares):

                    v_iou_matrix[i, j, k, l] = bb_intersection_over_union(
                        s1, s2, s3, s4
                    )

    best_fits = list(
        zip(
            *np.unravel_index(
                np.argsort(v_iou_matrix, axis=None, kind="stable"), v_iou_matrix.shape
            )
        )
    )
    best_fits.reverse()

    show_text(result, f"{len(v_squares)}, {len(h_squares)}", (20, 20), (0, 0, 0))

    min_iou = cv2.getTrackbarPos("Min IOU x10 000", "Trackbars") / 10_000
    count = 1

    found_squares = []
    max_iou = cv2.getTrackbarPos("Max IOU", "Trackbars") / 10_000

    for best_fit in best_fits:
        if v_iou_matrix[best_fit] < min_iou:
            break

        s1 = v_squares[best_fit[0]]
        s2 = v_squares[best_fit[1]]
        s3 = h_squares[best_fit[2]]
        s4 = h_squares[best_fit[3]]

        xA = min(s1[0], s2[0], s3[0], s4[0])
        yA = min(s1[1], s2[1], s3[1], s4[1])
        xB = max(s1[2], s2[2], s3[2], s4[2])
        yB = max(s1[3], s2[3], s3[3], s4[3])

        window = (xA, yA, xB, yB)

        # we check that our window is not similar to another already found window
        if found_squares == []:
            found_squares.append((v_iou_matrix[best_fit], window))
        else:
            valid = True
            for _, window2 in found_squares:
                if iou_2(window, window2) >= max_iou:
                    valid = False
                    break

            if valid:
                found_squares.append((v_iou_matrix[best_fit], window))

        # draw_rect(result, s1, (0, 255, 0))
        # draw_rect(result, s2, (0, 255, 0))
        # draw_rect(result, s3, (0, 255, 0))
        # draw_rect(result, s3, (0, 255, 0))

    for i, (score, window) in enumerate(found_squares):
        draw_rect(
            result,
            window,
            (0, int(255 * score), 0),
            thickness=2,
        )

        show_text(result, f"{score:.2f}", (250, 20 * (i + 1)), (0, 0, 0))

    show_text(result, f"{np.max(v_iou_matrix)}", (250, 300), (0, 0, 0))

    result = np.concatenate(
        (
            three_channels(result),
            three_channels(h_sides),
            three_channels(v_sides),
            three_channels(hist)
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
    cv2.createTrackbar("Filter Width", "Trackbars", 30, 50, empty)
    cv2.createTrackbar("Bar Width", "Trackbars", 20, 100, empty)

    cv2.createTrackbar("Min IOU x10 000", "Trackbars", 300, 10_000, empty)
    cv2.createTrackbar("Max IOU", "Trackbars", 4000, 10_000, empty)

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
