import numpy as np  
import cv2   
import glob
import math

def rotMat(R):
    alpha=math.atan((R[1][0])/(R[0][0]))
    beta=math.atan((-R[2][0])/math.sqrt((R[2][1])**2+(R[2][2])**2))
    gamma=math.atan((R[2][1])/(R[2][2]))
    return (alpha,beta,gamma)

#FOR CAMERA CALLIBRATION 

#DEFINE THE COORDINATES OF THE CORNER POINTS OF THE CHESSBOARD IN THE CHESSBOARD REFERENCE FRAME
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
objpoints = [] 
imgpoints = [] 

#ACCESSING THE IMAGES TO BE USED FOR CALLIBRATION
images = glob.glob('pics/*.jpg')

#CONDITION TO STOP CHECKING AFTER AN ACCURACY IS REACHED IN CORNER IDENTIFICATION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:

    #READ IMAGE AND FIND THE CORNERS OF THE CHESSBOARD USING INBUILT FUNCTIONS 
    img=cv2.imread(fname,cv2.IMREAD_GRAYSCALE)
    ret,corners=cv2.findChessboardCorners(img,(7,9),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +  cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret==True:

        #FOR EACH IMAGE, REFINE THE CORNERS AND THEN APPEND THE INTO IMAGE POINTS LIST
        objpoints.append(objp)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)  
        imgpoints.append(corners)

        #DRAW THE CHESSBOARD CORNERS USING INBUILT FUNCTION
        '''
        img = cv2.drawChessboardCorners(cv2.imread(fname), (9,7), corners,ret)
        cv2.imshow("CHESSBOARD", img)
        cv2.waitKey(500)
        '''

#USING THE INBUILT FUNCTION GENERATE THE CAMERA MATRIX AND THE DISTORTION MATRIX        
ret, Camera_matrix, Distortion_matrix, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2],None,None)

print("Camera Matrix:\n")
print(Camera_matrix)
print("\nDistortion Matrix:\n")
print(Distortion_matrix)

#SETTING ARUCO TAG DETECTOR PARAMETER TO DEFAULT AND ADDING REFINING CORNER METHOD
params =  cv2.aruco.DetectorParameters_create()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

font=cv2.FONT_HERSHEY_PLAIN

#CAPTURING VIDEO FROM WEBCAM
cap=cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#result = cv2.VideoWriter('filename.mp4', fourcc, 15, size)

while(True):

    #FETCHING THE IMAGE FRAME 
    ret, frame = cap.read()
    if ret is not True:
        print("Error in video capture")
        exit()

    #USING INBUILT FUNCTION TO DETECT MARKER AND GET THE CORNERS 
    corners, ids, rejectedImgPoints=cv2.aruco.detectMarkers(frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),parameters=params,cameraMatrix=Camera_matrix,distCoeff=Distortion_matrix)

    if ids is not None:

        #DRAWING THE BORDERS IN THE IMAGE AROUND THE ARUCO TAG
        (topLeft, topRight, bottomRight, bottomLeft) = corners[0][0]

        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        cv2.line(frame, topLeft, topRight, (0, 255, 0), 5)
        cv2.line(frame, topRight, bottomRight, (0, 255, 0), 5)
        cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 5)
        cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 5)

        #POSE ESTIMATION USING INBUILT FUNCTION TO GET THE ROTATION AND TRANSLATION VECTORS
        rvecs,tvecs,objp = cv2.aruco.estimatePoseSingleMarkers(corners[0], markerLength=10, cameraMatrix=Camera_matrix, distCoeffs=Distortion_matrix)

        #DRAW AXIS ON THE ARUCO TAG IN IMAGE 
        cv2.aruco.drawAxis(frame, cameraMatrix=Camera_matrix, distCoeffs=Distortion_matrix, rvec=rvecs, tvec=tvecs, length=5)

        #DISPLAY THE DISTANCE OF THE MARKER FROM CAMERA FRAME
        str_position = "Position X=%4.0f cm  Y=%4.0f cm Z=%4.0f cm"%(tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2])
        frame = cv2.putText(frame, text=str_position, org=(20, 40), fontFace=font, fontScale=2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        #DISPLAY THE YAW,PITCH,ROLL OF THE MARKER FROM CAMERA FRAME
        Rotation_matrix=cv2.Rodrigues(rvecs)[0]
        yaw,pitch,roll=rotMat(Rotation_matrix)
        str_rot = "Yaw = %4.0f   Pitch = %4.0f  Roll = %4.0f"%(math.degrees(yaw), math.degrees(pitch), math.degrees(roll))
        frame = cv2.putText(frame, text=str_rot, org=(20, 80), fontFace=font, fontScale=2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("RESULT", frame)
    #result.write(frame)

    k=cv2.waitKey(1)
    if k==27:

        cv2.destroyAllWindows()
        
        #result.release()
        cap.release()
        break
