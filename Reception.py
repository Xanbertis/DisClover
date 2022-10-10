import cv2, socket, numpy, pickle
import numpy as np
import numpy as np
from math import tan, pi

s=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
ip="192.168.43.74"
port=5000
s.bind((ip,port))

# Fonction vide pour les trackbars
def empty(a):
    pass


def getContours(image, areaMin):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if areaMin < area:
            cv2.drawContours(imageFiltered, cnt, -1, (0 , 0, 255), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            objectCorners = len(approx)
            x, y, width, height = cv2.boundingRect(approx)

            if objectCorners == 4:
                objectType = "Window"
            else:
                objectType = "None"
                return "none","none","none","none"

            if objectType == "Window":
                cv2.rectangle(imageFiltered, (x, y), (x + width, y + height), (0, 255, 0), 3)
                cv2.putText(imageFiltered, objectType, (x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            return x, y, width, height
    return "none","none","none","none"




"""
Returns R, T transform from src to dst
"""
def get_extrinsics():
    R = np.reshape([0.99998, -0.00207172, 0.00592144, 0.0021055, 0.999981, -0.00570433, -0.00590952, 0.00571668, 0.999966], [3,3]).T
    T = np.array([-0.0640291, -0.0003259, -0.00015527])
    return (R, T)

"""
Returns a camera matrix K from librealsense intrinsics
"""
def camera_matrix_left():
    return np.array([[286.286,             0, 418.365],
                     [            0, 286.478, 400.399],
                     [            0,             0,              1]])

def camera_matrix_right():
    return np.array([[285.268,             0, 428.706],
                     [            0, 285.263, 405.704],
                     [            0,             0,              1]])

"""
Returns the fisheye distortion from librealsense intrinsics
"""
def fisheye_distortion_left():
    return np.array([-0.007147947791963816, 0.0469416007399559, -0.04429551959037781, 0.008382633328437805])

def fisheye_distortion_right():
    return np.array([-0.005401404108852148, 0.03945644944906235, -0.03596261888742447, 0.0054514058865606785])

# Paramètres par défaut des valeurs minimales et maximales de couleurs (de 0 à 255) pour les filtrer dans l'image.
blueMin = 150
blueMax = 255
greenMin = 0
greenMax = 100
redMin = 0
redMax = 0

# Paramètres par défaut des fonctions d'érosion et de dilatation.
kernelValue = 15
kernel = np.ones((kernelValue, kernelValue), np.uint8)
iterations = 2

# Paramètres par défaut de l'air minimal et maximal du rectangle détecté (pour éviter de détecter des petits rectangles qui peuvent apparaîtrent dans l'image).
areaMin = 200
areaMax = 100000

# Valeurs basses et hautes pour le mask. On va garder les couleurs entre ces valeurs.
lower = np.array([blueMin, greenMin, redMin])
upper = np.array([blueMax, greenMax, redMax])


try:
    # Set up an OpenCV window to visualize the results
    # CIBLE SUR L'IMAGE PARAMETRABLE
    targetPointX = 800
    targetPointY = 540
    WINDOW_TITLE = "Target Point Position"

    # Configure the OpenCV stereo algorithm. See
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
    # description of the parameters
    window_size =0
    min_disp = 0
    # must be divisible by 16
    num_disp = 112 - min_disp
    max_disp = min_disp + num_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = 16,
                                   P1 = 8*3*window_size**2,
                                   P2 = 32*3*window_size**2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 32)

    # Translate the intrinsics from librealsense into OpenCV
    K_left  = camera_matrix_left()
    D_left  = fisheye_distortion_left()
    K_right = camera_matrix_right()
    D_right = fisheye_distortion_right()
    (width, height) = (848, 800)

    # Get the relative extrinsics between the left and right camera
    (R, T) = get_extrinsics()

    # We need to determine what focal length our undistorted images should have
    # in order to set up the camera matrices for initUndistortRectifyMap.  We
    # could use stereoRectify, but here we show how to derive these projection
    # matrices from the calibration and a desired height and field of view

    # We calculate the undistorted focal length:
    #
    #         h
    # -----------------
    #  \      |      /
    #    \    | f  /
    #     \   |   /
    #      \ fov /
    #        \|/
    stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
    stereo_height_px = 300          # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras
    R_left = np.eye(3)
    R_right = R

    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px + max_disp
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1)/2 + max_disp
    stereo_cy = (stereo_height_px - 1)/2


    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length
    P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])
    P_right = P_left.copy()
    P_right[0][3] = T[0]*stereo_focal_px

    # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
    # since we will crop the disparity later
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T[0], 0]])

    # Create an undistortion map for the left and right camera which applies the
    # rectification and undoes the camera distortion. This only has to be done
    # once
    m1type = cv2.CV_32FC1
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
    (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
    undistort_rectify = {"left"  : (lm1, lm2),
                         "right" : (rm1, rm2)}
    while True:
        x= s.recvfrom(1000000)
        data = x[0]
        data=pickle.loads(data)
        left = cv2.imdecode(data[0], cv2.IMREAD_GRAYSCALE)
        right = cv2.imdecode(data[1], cv2.IMREAD_GRAYSCALE)
        center_undistorted = {"left": cv2.remap(src=left,
                                                map1=undistort_rectify["left"][0],
                                                map2=undistort_rectify["left"][1],
                                                interpolation=cv2.INTER_LINEAR),
                              "right": cv2.remap(src=right,
                                                 map1=undistort_rectify["right"][0],
                                                 map2=undistort_rectify["right"][1],
                                                 interpolation=cv2.INTER_LINEAR)}
        # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
        disparity = stereo.compute(center_undistorted["left"], center_undistorted["right"]).astype(np.float32) / 16.0

        # re-crop just the valid part of the disparity
        disparity = disparity[min_disp:,:]
        cv2.imshow("disparity", disparity)
        # convert disparity to 0-255 and color it
        disp_vis = 255 * (disparity - min_disp) / num_disp
        #disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis, 1), cv2.COLORMAP_JET)
        color_image = cv2.cvtColor(center_undistorted["left"][:, max_disp:], cv2.COLOR_GRAY2RGB)
        #cv2.imshow("color",color_image)
        #image_left = center_undistorted["left"][min_disp:,:]
        #image_right = center_undistorted["right"][min_disp:,:]
        #img = cv2.hconcat([image_left,image_right])
        #image = center_undistorted["left"][:,max_disp:]
        mask = cv2.inRange(color_image, lower, upper)
        mask = cv2.dilate(mask, kernel, iterations=iterations)
        mask = cv2.erode(mask, kernel, iterations=iterations)

        #imageFiltered = cv2.bitwise_and(image, image, mask=mask)
        imageFiltered = cv2.bitwise_and(color_image, color_image, mask=mask)

        x, y, width, height = getContours(mask, areaMin)

        if x == "none":
            centerWinX = "none"
            centerWinY = "none"
        else:
            centerWinX = int(x) + int(width) // 2
            centerWinY = int(y) + int(height) // 2

        if x != "none":
            cv2.imwrite("Ressources/winOut.png", imageFiltered)

        messageX = ""
        messageY = ""
        targetPointXMin = imageFiltered.shape[1] // 2 - 10
        targetPointXMax = imageFiltered.shape[1] // 2 + 10
        targetPointYMin = imageFiltered.shape[0] // 2 - 10
        targetPointYMax = imageFiltered.shape[0] // 2 + 10

        if centerWinX == "none":
            messageX = "none"
        else:
            if int(centerWinX) < targetPointXMin:
                messageX = "GAUCHE"
            elif int(centerWinX) > targetPointXMax:
                messageX = "DROITE"
            else:
                messageX = "OK"

        if centerWinY == "none":
            messageY = "none"
        else:
            if int(centerWinY) < targetPointYMin:
                messageY = "HAUT"
            elif int(centerWinY) > targetPointYMax:
                messageY = "BAS"
            else:
                messageY = "OK"

        print("INSTRUCTIONS\tX : " + messageX + "\tY : " + messageY)
        if messageX == "OK" and messageY == "OK":
            print("AVANCER")

        cv2.imshow("Image de base", color_image)
        cv2.imshow("Mask", mask)
        cv2.imshow("Image filtrée et détection", imageFiltered)

        if cv2.waitKey(10) == 13:
            break
finally:
    cv2.destroyAllWindows()