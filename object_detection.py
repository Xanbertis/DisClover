import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


from utils import empty, show_text

UNKNOWN = 0
YELLOW = 1
BLUE = 2
RED = 3


def find_color(c, image_hsv, noise_hsv, line):
    center = (c[0], c[1])
    radius = c[2]

    mask = np.zeros(image_hsv.shape[:2], np.uint8)

    cv2.circle(
        mask, center, int(radius + 0.1 * radius), (255, 255, 255), thickness=cv2.FILLED
    )

    cv2.circle(
        mask, center, int(radius - 0.1 * radius), (0, 0, 0), thickness=cv2.FILLED
    )

    inverted_mask = cv2.bitwise_not(mask)

    normal = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    noisy = cv2.bitwise_and(noise_hsv, noise_hsv, mask=inverted_mask)

    result = cv2.bitwise_or(noisy, normal)

    # now to crop the result

    top_x = max(0, int(center[1] - 1.1 * radius))
    top_y = max(0, int(center[0] - 1.1 * radius))

    bot_x = min(image_hsv.shape[0], int(center[1] + 1.1 * radius))
    bot_y = min(image_hsv.shape[1], int(center[0] + 1.1 * radius))

    result = result[top_x:bot_x, top_y:bot_y]

    histr = cv2.calcHist([result], [0], None, [256], [0, 256]) / (
        np.prod(result.shape[:2]) / 3
    )
    line.set_ydata(histr)

    argmax = np.argmax(histr)

    if (argmax >= 170 and argmax <= 180) or argmax <= 3:
        return RED
    elif argmax >= 19 and argmax <= 30:
        return YELLOW
    elif argmax >= 105 and argmax <= 115:
        return BLUE
    else:
        return UNKNOWN


def main():

    kernel = np.ones((5, 5), np.uint8)

    # capture camera 0
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Saturation")
    cv2.namedWindow("Output")

    cv2.createTrackbar("Lower", "Saturation", 185, 255, empty)
    cv2.createTrackbar("Upper", "Saturation", 255, 255, empty)

    cv2.createTrackbar("dp", "Output", 7, 15, empty)
    cv2.createTrackbar("MinDist", "Output", 100, 500, empty)
    cv2.createTrackbar("Param1", "Output", 100, 500, empty)
    cv2.createTrackbar("Param2", "Output", 225, 1000, empty)

    noise_hsv = np.zeros(
        (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3,
        ),
        np.uint8,
    )
    cv2.randu(noise_hsv, (0, 0, 0), (180, 256, 256))

    fig, ax = plt.subplots()
    (line,) = ax.plot(np.arange(256), np.zeros((256, 1)), c="k", lw=3)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 0.4)
    plt.ion()
    plt.show()

    start_time = perf_counter()
    fps = 30

    while True:

        current_time = perf_counter()
        duration = current_time - start_time
        start_time = current_time

        new_fps = 1 / duration

        if abs(new_fps - fps) >= 2:
            fps = new_fps

        ret, image = cap.read()
        output_frame = image.copy()

        show_text(output_frame, f"FPS: {fps:.2f}", (10, 20), (203, 192, 255))

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV, dstCn=1)
        sat_channel = image_hsv[:, :, 1]

        sat_channel = cv2.bilateralFilter(sat_channel, 5, 200, 200)
        sat_channel = cv2.erode(sat_channel, kernel, iterations=1)
        # sat_channel = cv2.dilate(sat_channel, kernel, iterations=2)

        lower = cv2.getTrackbarPos("Lower", "Saturation")
        upper = cv2.getTrackbarPos("Upper", "Saturation")

        mask = cv2.inRange(sat_channel, lower, upper)
        sat_channel = cv2.bitwise_and(sat_channel, sat_channel, mask=mask)

        cv2.imshow("Saturation", sat_channel)

        cv2.imshow("Hue", image_hsv[:, :, 0])

        dp = cv2.getTrackbarPos("dp", "Output")
        min_dist = cv2.getTrackbarPos("MinDist", "Output")
        param1 = cv2.getTrackbarPos("Param1", "Output")
        param2 = cv2.getTrackbarPos("Param2", "Output")

        black_filter = cv2.inRange(image_hsv[:, :, 2], lower, upper)
        val_channel = cv2.bitwise_and(
            image_hsv[:, :, 2], image_hsv[:, :, 2], mask=black_filter
        )

        cv2.imshow("Value", val_channel)

        circles = cv2.HoughCircles(
            val_channel,
            cv2.HOUGH_GRADIENT,
            dp,
            min_dist,
            param1=param1,
            param2=param2,
        )

        if circles is not None:

            circles = np.uint16(np.around(circles))[0, :]

            for i, c in enumerate(circles):

                # need to find the color of the circle
                color = find_color(c, image_hsv, noise_hsv, line)

                if color == RED:
                    color = (0, 0, 255)
                elif color == YELLOW:
                    color = (0, 255, 255)
                elif color == BLUE:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 0)

                center = (c[0], c[1])
                cv2.circle(output_frame, center, c[2], color, thickness=2)
                cv2.circle(output_frame, center, 1, color, thickness=2)

        cv2.imshow("Output", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
