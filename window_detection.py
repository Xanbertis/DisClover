import cv2
import numpy as np

from utils import downscale


def main():

    color_image = cv2.imread("./images/window_1.jpg")

    color_image = downscale(color_image, 5, interpolation=cv2.INTER_CUBIC)

    target_point = (640 // 2, 480 // 2)

    while True:

        image = color_image.copy()

        result = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        result = cv2.inRange(result, np.array([0, 0, 0]), np.array([180, 255, 30]))

        canny = cv2.erode(result, (7, 7))
        canny = cv2.dilate(canny, (7, 7))

        max_area = 5000
        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

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
                        print(f"{aspect_ratio = }")
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

        # cv2.imshow("grey", color_image)
        cv2.imshow("color image", image)
        cv2.imshow("canny", canny)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
