# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        
        # return the edged image
        return edged

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
imagePaths = list(paths.list_images(args["images"]))

for imagePath in imagePaths:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(4, 4), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
                pad_w, pad_h = int(0.15*w), int(0.075*h)
		cv2.rectangle(orig, (x+pad_w, y+pad_h), (x + w-pad_w, y + h - pad_h), (0, 0, 255), 2)
                #print(x + pad_w, y + pad_h, x + w - pad_w, y + h - pad_h)
                head = y + pad_h
                print(head)

	filename = imagePath[imagePath.rfind("/") + 1:]

        # door detection
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        auto = auto_canny(blurred)
        dst = cv2.cornerHarris(auto, 3, 7, 0.1)

#        cv2.imshow('dst', auto)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
        # add color back to the black and white image
        auto = cv2.cvtColor(auto, cv2.COLOR_GRAY2BGR)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        auto[dst>0.001*dst.max()]=[0,0,255]

        # Get index of detected corner points
        idx = dst>0.001*dst.max()
        # Finding the first corner found at the top of the image : top door frame
        top = np.argwhere(idx==True)[0][0]
        # Now take the column of this point, and go down in the column to find the bottom of the door frame
        look_col = np.argwhere(idx==True)[0][1]

        # Pad that column to take slight crookedness into account
        pad = 5
        bottom = np.argwhere(idx[:,look_col-pad:look_col+pad]==True)
        # Start the search for the bottom part from the middle of the image onwards
        bottom = bottom[bottom>len(idx)/2][0]

        door_pixel = bottom-top
        door_cm = 227.0
        
        child_pixel = bottom-head

        # Calculate the child's heigth from the doors pixels
        child_cm = (door_cm/door_pixel)*child_pixel
        print(child_cm)

        cv2.imshow('dst', auto)

	# show the output images
	cv2.imshow("Before NMS", orig)
        cv2.imwrite('proc_gif/'+filename,orig)
	cv2.waitKey(0)


