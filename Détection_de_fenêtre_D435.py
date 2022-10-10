import cv2
import numpy as np
import pyrealsense2 as rs


#Préparation du flux vidéo Realsense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
pipeline.start(config)
colorizer = rs.colorizer()

targetPointX = 640 // 2
targetPointY = 480 // 2

while True:
    #Récupération des données vidéos
    frames = pipeline.wait_for_frames()
    color_frames = frames.get_color_frame()
    #Récupération des données vidéos en format lisible par OpenCV
    color_image = np.asanyarray(color_frames.get_data())
    image = color_image.copy()
    # converting to LAB color space
    #lab = cv2.cvtColor(color_image,cv2.COLOR_BGR2LAB)
    #l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    #cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    #limg = cv2.merge((cl, a, b))
    # Converting image from LAB Color model to BGR color spcae
    #enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #cv2.imshow('enhanced',enhanced_img)
    result = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    #color_image = cv2.cvtColor(enhanced_img,cv2.COLOR_RGB2GRAY)

    #color_image = cv2.GaussianBlur(color_image,(5,5),1)
    #color_image = cv2.filter2D(color_image,-1,kernel)

    '''
    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(color_image, cv2.MORPH_GRADIENT, kernel)
    cv2.dilate(gradient,(5,5))
    ret, th = cv2.threshold(color_image, 0, 255, cv2.THRESH_OTSU)
    ret3, th3 = cv2.threshold(gradient, 0, 255, cv2.THRESH_OTSU)
    result = cv2.bitwise_and(color_image,th3)
    canny = cv2.Canny(gradient,75,75,4)
    result = cv2.goodFeaturesToTrack(canny,4,0.5,50)
    th = cv2.adaptiveThreshold(color_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    '''
    #color_image = cv2.morphologyEx(color_image,cv2.MORPH_TOPHAT,(9,9))
    #cv2.imshow('mdif',color_image)

    result = cv2.inRange(result,np.array([0,0,0]),np.array([180, 255, 30]))
    '''np.array([0,0,0]),np.array([180, 255, 30])'''
    #canny = cv2.morphologyEx(result,cv2.MORPH_CLOSE,(9,9))
    canny = cv2.erode(result,(7,7))
    canny = cv2.dilate(canny,(7,7))
    maxArea = 5000
    contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.circle(image, (targetPointX,targetPointY), 10, (0, 0, 0), 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > maxArea:
            perimeter = cv2.arcLength(cnt,True)

            approx = cv2.approxPolyDP(cnt,0.01*perimeter,True)
            objectCorners = len(approx)

            x, y, width, height = cv2.boundingRect(approx)

            if objectCorners == 4:
                aspRatio = width / float(height)
                aspRatio = np.round(aspRatio,3)
                if aspRatio > 0.85 and aspRatio < 1.15:
                    print(aspRatio)
                    cv2.circle(image, (int(x) + int(width) // 2, int(y) + int(height) // 2), 10, (0, 0, 0), 2)
                    objectType = "Square"
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)
                    #cv2.putText(image, objectType, (x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                     #       (0, 255, 0), 2)
                    cv2.putText(image, str(str(width)+" px,"+str(height)+" px"),(x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)
                    if x == "none":
                        centerWinX = "none"
                        centerWinY = "none"
                    else:
                        centerWinX = int(x) + int(width) // 2
                        centerWinY = int(y) + int(height) // 2

                    cv2.putText(image, "Window position : ( " + str(centerWinX) + " ; " + str(centerWinY) + " )",
                                (10, image.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    messageX = ""
                    messageY = ""
                    targetPointXMin = image.shape[1] // 2 - 10
                    targetPointXMax = image.shape[1] // 2 + 10
                    targetPointYMin = image.shape[0] // 2 - 10
                    targetPointYMax = image.shape[0] // 2 + 10

                    if centerWinX == "none":
                        messageX = "none"
                    else:
                        if int(centerWinX) < targetPointXMin:
                            messageX = "DROITE"
                        elif int(centerWinX) > targetPointXMax:
                            messageX = "GAUCHE"
                        else:
                            messageX = "OK"

                    if centerWinY == "none":
                        messageY = "none"
                    else:
                        if int(centerWinY) < targetPointYMin:
                            messageY = "BAS"
                        elif int(centerWinY) > targetPointYMax:
                            messageY = "HAUT"
                        else:
                            messageY = "OK"

                    print("INSTRUCTIONS\tX : " + messageX + "\tY : " + messageY)
                    if messageX == "OK" and messageY == "OK":
                        cv2.circle(image, (targetPointX, targetPointY), 10, (255, 255, 255), 2)
                        print("AVANCER")
                '''else:
                    objectType = "No Square"
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3)
                    cv2.putText(image, objectType, (x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0,0,255), 2)
            else:
                objectType = "No square/rectangle"
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3)
                cv2.putText(image, objectType, (x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)'''

    #cnt = contours[4]
    #cv2.drawContours(color_image, [cnt], 0, (0, 255, 0), 3)
    #depth_image = cv2.bitwise_and(canny,depth_image)
    cv2.imshow('grey',color_image)
    cv2.imshow('color image', image)
    #cv2.imshow('gdrt', gradient)
    cv2.imshow('canny', canny)
    #cv2.imshow('colored depth', depth_image)
    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()
        cv2.destroyAllWindows()
        break