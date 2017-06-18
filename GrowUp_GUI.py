# USAGE
# python detect.py --images images
# import the necessary packages

from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import boto3
import numpy
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from datetime import datetime
import tkinter
from tkinter import *
from PIL import ImageTk, Image


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
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
#imagePaths = list(paths.list_images(args["images"]))

def kidsgrowth(imagePath):
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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
        pad_w, pad_h = int(0.15 * w), int(0.075 * h)
        cv2.rectangle(orig, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 0, 255), 2)
        # print(x + pad_w, y + pad_h, x + w - pad_w, y + h - pad_h)
        head = y + pad_h

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
    auto[dst > 0.001 * dst.max()] = [0, 0, 255]

    # Get index of detected corner points
    idx = dst > 0.001 * dst.max()
    # Finding the first corner found at the top of the image : top door frame
    top = np.argwhere(idx == True)[0][0]
    # Now take the column of this point, and go down in the column to find the bottom of the door frame
    look_col = np.argwhere(idx == True)[0][1]

    # Pad that column to take slight crookedness into account
    pad = 5
    bottom = np.argwhere(idx[:, look_col - pad:look_col + pad] == True)
    # Start the search for the bottom part from the middle of the image onwards
    bottom = bottom[bottom > len(idx) / 2][0]

    door_pixel = bottom - top
    door_cm = 227.0

    child_pixel = bottom - head

    # Calculate the child's heigth from the doors pixels
    child_cm = (door_cm / door_pixel) * child_pixel
    print(child_cm)

    cv2.imshow('dst', auto)

    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imwrite('proc_gif/' + filename, orig)
    cv2.waitKey(0)

    return (child_cm)

# create root GUI class
class main_GUI:
    def __init__(self, master):
        self.master = master
        master.title("main")

        self.birthday = list()

        # graph creation method
        def create_graph():
            Fig = Figure(figsize = (5, 5), dpi = 100)
            a = Fig.add_subplot(111)
            a.plot(self.dd, self.h)

            canvas = FigureCanvasTkAgg(Fig, master = self.master)
            canvas.show()
            canvas.get_tk_widget().pack(side= TOP, fill= BOTH, expand=1)

        # method for adding and evaluating a new image
        def add_image():
            def get_date():
                dd = add_console.date_entry.get()
                dd = datetime(year = int(dd[0:4]), month = int(dd[4:6]), day = int(dd[6:8]))
                self.dd.append(dd)
                #img = ImageTk.PhotoImage(Image.open(add_console.path_entry.get()))
                path = (add_console.path_entry.get())

                tmp = kidsgrowth(path)
                self.h.append(tmp)
                add_console.destroy()

            add_console = tkinter.Toplevel()
            add_console.title("add_image")

            add_console.date_frame = tkinter.LabelFrame(add_console, text = "date (YYYYMMDD): ")
            add_console.date_frame.pack(side=LEFT)
            add_console.date_entry = tkinter.Entry(add_console.date_frame)
            add_console.date_entry.pack()

            add_console.path_frame = tkinter.LabelFrame(add_console, text="image path: ")
            add_console.path_frame.pack(side=RIGHT)
            add_console.path_entry = tkinter.Entry(add_console.path_frame)
            add_console.path_entry.pack()

            add_console.enter_button_frame = tkinter.LabelFrame(add_console)
            add_console.enter_button_frame.pack(side=BOTTOM)
            add_console.enter_button = tkinter.Button(add_console.enter_button_frame, text="enter", command=get_date)
            add_console.enter_button.pack()

        # method for adding birthday (currently out of use)
        def add_birthday():
            def get_bday():
                birthday = entry.get()
                birthday = datetime(
                    year=int(birthday[0:4]), month=int(birthday[4:6]), day=int(birthday[6:8])
                )
                self.birthday.append(birthday)
                add_bd.destroy()

            add_bd = tkinter.Toplevel()
            add_bd.title("add_bd")

            entry = tkinter.Entry(add_bd)
            entry.pack()
            tkinter.Button(add_bd, text = "enter birthdate (YYYYMMDD", command = get_bday).pack()

        #add_birthday()

        # labels and buttons for root GUI
        self.label = tkinter.Label(master, text = "kidsgrowth GUI")
        self.label.pack()

        self.new_image_button = tkinter.Button(master, text = "Add a new image", command = add_image)
        self.new_image_button.pack(side = LEFT)

        self.close_button = tkinter.Button(master, text = "Exit", command = master.quit)
        self.close_button.pack(side = RIGHT)

        self.graph_button = tkinter.Button(master, text = "create graph", command = create_graph)
        self.graph_button.pack()

        # date and height lists
        self.dd = list()
        self.h = list()



root = Tk()
my_gui = main_GUI(root)
root.mainloop()
