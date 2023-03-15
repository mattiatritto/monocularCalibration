import numpy as np
import cv2 as cv
import glob



# These are the number of squares that are on the chessboard used for calibration
numMinSquares = 23;
numMaxSquares = 31;

# This line sets the end of the algorithm, max 30 iterations OR 0.001 error
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Object points are points in real 3D world (Example: ((0,0,0), (0,0,1), ..., (4, 5, 9), ...)
objectPoints = np.zeros((numMinSquares*numMaxSquares,3), np.float32)

# ????
objectPoints[:,:2] = np.mgrid[0:numMaxSquares,0:numMinSquares].T.reshape(-1,2)

# These arrays are used to store object points and image points from all the images
# 3D points in real world space
objectPointsoints = []
# 2D points in image plane
imgpoints = []



# Takes all the images that have .jpg as extensions
images = glob.glob('*.jpg')
print("Loading " + str(len(images)) + " images ...");



# For every image...
for fname in images:

    # Loads images and trasnform in black & white
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (numMaxSquares,numMinSquares), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objectPointsoints.append(objectPoints)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, (numMaxSquares,numMinSquares), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()



# This returns the camera matrix, distortion coefficients, rotation and translation vectors
ret, cameraMatrix, distorsionCoefficients, rotationVector, translationVector = cv.calibrateCamera(objectPointsoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix: ");
print(cameraMatrix);

print("Distorsion coefficients: ");
print(distorsionCoefficients);

print("Rotation vector: ");
print(rotationVector);

print("Translation vector: ");
print(translationVector);



# Undistorsion of a single picture
img = cv.imread('camera_calib5.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distorsionCoefficients, (w,h), 1, (w,h))
dst = cv.undistort(img, cameraMatrix, distorsionCoefficients, None, newcameramtx)
# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('imageUndistorted.png', dst)



# Estimate the re-projection error 
mean_error = 0
for i in range(len(objectPointsoints)):
    imgpoints2, _ = cv.projectPoints(objectPointsoints[i], rotationVector[i], translationVector[i], cameraMatrix, distorsionCoefficients)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "Re-projection error: {}".format(mean_error/len(objectPointsoints)))

