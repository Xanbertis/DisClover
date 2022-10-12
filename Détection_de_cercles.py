from typing import Tuple
import numpy as np
import cv2

from utils import ColorDetector, show_text

# Matrice de taille 5*5, elle permet de faire de la convolution sur notre image pour en modifier ses caractéristiques
kernel = np.ones((5, 5), np.uint8) / 25

# Fonction de détection d'un cercle en détectant les contours


def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            objectCorners = len(approx)
            # On considère la forme comme un cercle à partir de 5 coins, et on retourne ses informations
            # print(objectCorners)
            if objectCorners > 10:
                return "OK"
    # Retour par défaut si aucun cercle n'a été trouvé
    return "NOT OK"


def close_enough(circle, other, epsilon=0.1):
    def proximity_ratio(a, b):
        return abs(1 - a / b)

    if (
        proximity_ratio(circle[0], other[0]) <= epsilon
        and proximity_ratio(circle[1], circle[2]) <= epsilon
        # and proximity_ratio(circle[2], other[2]) <= epsilon
    ):
        return True

    return False


def choose_action(black_circles, color_circles, target_point: Tuple[int, int], canvas):

    # print(color_circles)
    for i, circles in enumerate(color_circles):
        if circles is not None:
            color_circles[i] = np.uint16(np.around(circles))

    black_circles = np.uint16(np.around(black_circles))

    for c in black_circles[0, :]:

        out = False
        # iterate over all circles of all colors
        for circles in color_circles:

            if circles is not None:
                for other in circles[0, :]:
                    if close_enough(c, other):
                        out = True

                        center = (c[0], c[1])
                        cv2.circle(canvas, center, 1, (0, 255, 0), thickness=2)
                        cv2.circle(canvas, center, c[2], (0, 255, 0), thickness=2)

            if out:
                break

    """cv2.circle(
        canvas,
        center=(circles[0, 0], circles[0, 1]),
        radius=circles[0, 2],
        color=color,
        thickness=2,
    )
    cv2.circle(
        canvas,
        center=(circles[0, 0], circles[0, 1]),
        radius=2,
        color=color,
        thickness=2,
    )"""

    """if circles[0, 0] < target_point[0] - 10:
        messageX = "GAUCHE"
    elif target_point[0] + 10 < circles[0, 0]:
        messageX = "DROITE"
    else:
        messageX = "OK"

    if circles[0, 1] < target_point[1] - 10:
        messageY = "DEVANT"
    elif target_point[1] + 10 < circles[0, 1]:
        messageY = "DERRIERE"
    else:
        messageY = "OK"
        # Si le cercle est sur la cible
    if messageX == "OK" and messageY == "OK":
        show_text(canvas, "Cible atteinte", target_point, (255, 255, 255))
        # Affichage de la position du cercle dans l'image
        show_text(
            canvas,
            "Circle Position : (" + str(circles[0, 0]) +
            ";" + str(circles[0, 1]) + ")",
            (canvas.shape[1] - 250, canvas.shape[0] - 6),
            (255, 255, 255),
        )
        # print("INSTRUCTIONS\tDESCENTE")
    else:
        # Affichage de la position du cercle dans l'image
        show_text(
            canvas,
            "Circle Position : (" + str(circles[0, 0]) +
            ";" + str(circles[0, 1]) + ")",
            (canvas.shape[1] - 250, canvas.shape[0] - 6),
            color,
        )
        cv2.circle(canvas, target_point, 10, (0, 0, 0), 2)
        # Envoi des instructions dans la console
        #print("INSTRUCTIONS\tX : " + messageX + " Y : " + messageY)

        # Affichage de la position de la cible dans l'image
        show_text(
            canvas,
            "Target Position : ("
            + str(target_point[0])
            + ";"
            + str(target_point[1])
            + ")",
            (1, canvas.shape[0] - 6),
            color,
        )"""


def show_circles(circles, color, canvas):
    if circles is not None:

        circles = np.uint16(np.around(circles))

        for c in circles[0, :]:
            center = (c[0], c[1])
            cv2.circle(canvas, center, 1, color, thickness=2)
            cv2.circle(canvas, center, c[2], color, thickness=2)


def find_circles(color_detector, hsv_image, show_edges=True):
    filtered_image = cv2.inRange(
        hsv_image, color_detector.lower_bounds, color_detector.upper_bounds
    )

    # On applique du flou sur l'image pour réduire davantage le bruit
    filtered_image = cv2.GaussianBlur(filtered_image, (9, 9), 2, 2)

    # Cette fonction compare la différence entre deux éléments
    # Dans notre cas, nous appliquons un masque sur la comparaison pour ne faire ressortir uniquement la couleur associé à ce masque
    filtered_image_visu = cv2.bitwise_and(hsv_image, hsv_image, mask=filtered_image)
    filtered_image_visu = cv2.cvtColor(filtered_image_visu, cv2.COLOR_BGR2GRAY)

    l = cv2.getTrackbarPos("LowerThreshold", "Trackbar")
    h = cv2.getTrackbarPos("UpperThreshold", "Trackbar")

    # Canny edge detection algorithm
    image_edges = cv2.Canny(filtered_image_visu, h, l)

    image_edges = cv2.dilate(image_edges, np.ones((5, 5), np.uint8), iterations=1)

    if show_edges:
        cv2.imshow("Edges : " + color_detector.name, image_edges)

    # On applique la fonction HoughCircles pour détecter les cercles dans l'image, la fonction renvoit les coordonnées du centre du cercle ainsi que sa taille en pixel
    # La fonction fonctionne sur des images en grayscale ce qui est le cas de mes masques qui font ressortir en blanc les éléments que l'on veut détecter
    circles = cv2.HoughCircles(
        image_edges,
        cv2.HOUGH_GRADIENT,
        4.5,
        100,
        param1=100,
        param2=250,
        minRadius=20,
        maxRadius=500,
    )

    return circles


def empty(a):
    pass


class ColorCircles:
    def __init__(self, detector: ColorDetector, display_color: Tuple[int, int, int]):
        self.detector = detector
        self.display_color = display_color
        self.circles: list = []

    def update(self):
        self.detector.update_bounds()

    def find_circles(self, hsv_image):
        self.circles = find_circles(self.detector, hsv_image, True)


#    def choose_action(self, target, canvas):
#        # getContours was removed from check, need to check if it’s necessary or can be bypassed
#        if self.circles is not None:
#            choose_action(self.circles, target, canvas, self.display_color)


def main():
    # cap = cv2.VideoCapture('http://user:pass@192.168.43.190:8000/stream.mjpg')
    cap = cv2.VideoCapture(0)

    # CIBLE SUR L'IMAGE PARAMETRABLE
    target_point = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2),
    )

    # Trackbar avec slider
    cv2.namedWindow("Trackbar")
    cv2.resizeWindow("Trackbar", (300, 900))

    # Possible blue threshold: [20, 115, 70][255, 145, 120]
    blue_circles = ColorCircles(
        ColorDetector("Blue", "Trackbar", (106, 127, 126), (131, 216, 250)), (255, 0, 0)
    )
    # No other recorded possible value for red
    red_circles = ColorCircles(
        ColorDetector("Red", "Trackbar", (159, 50, 70)), (0, 0, 255)
    )
    # Possible yellow threshold: [20, 110, 170][20, 110, 170]
    yellow_circles = ColorCircles(
        ColorDetector("Yellow", "Trackbar", (25, 50, 70), (35, 255, 255)), (0, 255, 255)
    )

    black_circles = ColorCircles(
        ColorDetector("Black", "Trackbar", upper_bounds=(180, 255, 45)), (255, 255, 255)
    )

    circles = [black_circles, blue_circles]

    cv2.createTrackbar("LowerThreshold", "Trackbar", 75, 1000, empty)
    cv2.createTrackbar("UpperThreshold", "Trackbar", 600, 1000, empty)

    while True:
        # Capture frame-by-frame
        ret, image = cap.read()
        output_frame = image.copy()
        # print(h_min,h_max,s_min,s_max,v_min,v_max)

        kernel = np.ones((5, 5), np.uint8) / 25

        # FPS de la vidéo
        show_text(
            output_frame,
            "FPS : " + str(cap.get(cv2.CAP_PROP_FPS)),
            (image.shape[1] - 100, 15),
            (255, 0, 255),
        )

        image_bf = image.copy()
        # image_bf = cv2.bilateralFilter(image, 5, 400, 400)

        cv2.imshow("filtered", image_bf)

        # On convertit dans le format HSV pour pouvoir toucher à la teinte, la saturation et la luminosité de la couleur
        image_hsv = cv2.cvtColor(image_bf, cv2.COLOR_BGR2HSV)

        image_hsv = cv2.bilateralFilter(image_hsv, 5, 200, 200)
        image_hsv = cv2.erode(image_hsv, kernel, iterations=1)
        image_hsv = cv2.dilate(image_hsv, kernel, iterations=2)

        cv2.imshow("raw", image_hsv)

        # On désigne le milieu de l'image comme étant la cible à atteindre
        cv2.circle(output_frame, target_point, 1, (0, 0, 0), 3)

        for c in circles:
            c.update()
            c.find_circles(image_hsv)

            show_circles(c.circles, c.display_color, output_frame)

        if circles[0].circles is not None:
            choose_action(
                circles[0].circles,
                [circles[1].circles],
                target_point,
                output_frame,
            )

        cv2.imshow("frame", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # On stoppe et ferme le flux vidéo
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
