import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
    
    #Frame
def rescaleFrame(frame,scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions=(width,height)
    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)

def processImage (image):
    #Loading test images
    #For using live camera
    _, image = video_cap.read()

    #Convert to Grey Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Smoothing filter parametres
    gaussgray = cv2.GaussianBlur(gray,(3,3),0)
    
    # Applying Canny Edge Filter and Parameters
    cannygaussgray = cv2.Canny(gaussgray,50,50)
    
    mask = np.zeros_like(cannygaussgray)
    ignore_mask_color = 255
    
    # Region of Interest
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(230, 120), (490, 220), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(cannygaussgray, mask)
    
    # Hough Transform Parameters
    rho = 2 # Distance Resolution in pixels
    theta = np.pi/180 # Angular Resolution
    threshold = 15     # Threshold value
    min_line_length = 10 # Minimum number of pixels to draw a line
    max_line_gap = 30    # Maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # Creating a blank to draw lines
    
    # Runs Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((cannygaussgray, cannygaussgray, cannygaussgray))
    
    # Draw the lines on the original image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    gaussgray2 = cv2.GaussianBlur(lines_edges,(3,3),0)
    
    return lines_edges, gaussgray2, cannygaussgray

# To stream the image frame   
video_cap=cv2.VideoCapture(-1)
# Real Time outputs
while video_cap.isOpened():
    ret, frame = video_cap.read()
    if ret:
        # Naming of the filtered frames
        cikis, cikis2, cikis3 = processImage(frame)
        cikis2_resized = rescaleFrame(cikis2,scale=.75)
        cikis3_resized = rescaleFrame(cikis3,scale=.5)
      
        ##cv2.imshow('frame1',cikis2_resized)
        # All filtered applied and lines are visible
        cv2.imshow('frame2',cikis3_resized)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()