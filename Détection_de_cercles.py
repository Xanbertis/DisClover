from typing import Tuple
import numpy as np
import cv2

# Matrice de taille 5*5, elle permet de faire de la convolution sur notre image pour en modifier ses caractéristiques
kernel = np.ones((5, 5), np.uint8) / 25

# Fonction de détection d'un cercle en détectant les contours


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
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)


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


def choose_action(
    circles, target_point: Tuple[int, int], canvas, color: Tuple[int, int, int]
):
    circles = np.round(circles[0, :]).astype("int")
    cv2.circle(
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
    )

    if circles[0, 0] < target_point[0] - 10:
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
            "Circle Position : (" + str(circles[0, 0]) + ";" + str(circles[0, 1]) + ")",
            (canvas.shape[1] - 250, canvas.shape[0] - 6),
            (255, 255, 255),
        )
        print("INSTRUCTIONS\tDESCENTE")
    else:
        # Affichage de la position du cercle dans l'image
        show_text(
            canvas,
            "Circle Position : (" + str(circles[0, 0]) + ";" + str(circles[0, 1]) + ")",
            (canvas.shape[1] - 250, canvas.shape[0] - 6),
            color,
        )
        cv2.circle(canvas, target_point, 10, (0, 0, 0), 2)
        # Envoi des instructions dans la console
        print("INSTRUCTIONS\tX : " + messageX + " Y : " + messageY)

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
        )


def find_circles(color_detector, hsv_image, show_edges=True):
    filtered_image = cv2.inRange(
        hsv_image, color_detector.lower_bounds, color_detector.upper_bounds
    )

    # On applique du flou sur l'image pour réduire davantage le bruit
    filtered_image = cv2.GaussianBlur(filtered_image, (9, 9), 2, 2)

    # Cette fonction compare la différence entre deux éléments
    # Dans notre cas, nous appliquons un masque sur la comparaison pour ne faire ressortir uniquement la couleur associé à ce masque
    filtered_image_visu = cv2.bitwise_and(hsv_image, hsv_image, mask=filtered_image)

    # Canny edge detection algorithm
    image_edges = cv2.Canny(filtered_image_visu, 75, 75)

    if show_edges:
        cv2.imshow("Edges : " + color_detector.name, image_edges)

    # On applique la fonction HoughCircles pour détecter les cercles dans l'image, la fonction renvoit les coordonnées du centre du cercle ainsi que sa taille en pixel
    # La fonction fonctionne sur des images en grayscale ce qui est le cas de mes masques qui font ressortir en blanc les éléments que l'on veut détecter
    circles = cv2.HoughCircles(
        image_edges,
        cv2.HOUGH_GRADIENT,
        hsv_image.shape[0] / 64,
        hsv_image.shape[0] / 8,
        param1=100,
        param2=30,
        minRadius=0,
        maxRadius=0,
    )

    return circles


def empty(a):
    pass


class ColorCircles:
    def __init__(self, detector: ColorDetector, display_color: Tuple[int, int, int]):
        self.detector = detector
        self.display_color = display_color
        self.circles = []

    def update(self):
        self.detector.update_bounds()

    def find_circles(self, hsv_image):
        self.circles = find_circles(self.detector, hsv_image, True)

    def choose_action(self, target, canvas):
        # getContours was removed from check, need to check if it’s necessary or can be bypassed
        if self.circles is not None:
            choose_action(self.circles, target, canvas, self.display_color)


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
        ColorDetector("Blue", "Trackbar", (90, 50, 70), (128, 255, 255)), (255, 0, 0)
    )
    # No other recorded possible value for red
    red_circles = ColorCircles(
        ColorDetector("Red", "Trackbar", (159, 50, 70)), (0, 0, 255)
    )
    # Possible yellow threshold: [20, 110, 170][20, 110, 170]
    yellow_circles = ColorCircles(
        ColorDetector("Yellow", "Trackbar", (25, 50, 70), (35, 255, 255)), (0, 255, 255)
    )
    circles = [blue_circles, red_circles, yellow_circles]

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

        # Cette fonction permet de réduire le bruit sur l'image
        image_bgr = cv2.bilateralFilter(image, 5, 200, 200)
        # L'érosion et la dilation permettent de volontairement dégrader l'image afin d'en faire ressortir des éléments
        image_bgr = cv2.erode(image_bgr, kernel, iterations=3)
        image_bgr = cv2.dilate(image_bgr, kernel, iterations=5)
        cv2.imshow("erode", image_bgr)
        # On convertit dans le format HSV pour pouvoir toucher à la teinte, la saturation et la luminosité de la couleur
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        cv2.imshow("raw", image_hsv)

        # On désigne le milieu de l'image comme étant la cible à atteindre
        cv2.circle(output_frame, target_point, 1, (0, 0, 0), 3)

        for c in circles:
            c.update()
            c.find_circles(image_hsv)
            c.choose_action(target_point, output_frame)

        cv2.imshow("frame", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # On stoppe et ferme le flux vidéo
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
