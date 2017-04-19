import cv2
import sys
import logging as log
import numpy as np

from time import sleep


def face_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


class HandTrack():

    def __init__(self):
        self.fgmask = None
        # self.fgbg = cv2.bgsegm.createBackgroundSubtractor()
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    def get_background(self, frame):
        if self.fgmask is None:
            self.fgmask = self.fgbg.apply(frame)
            print('Got background')
    #    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    def remove_bg(self, frame):
        if self.fgmask is not None:
            fg_mask = self.fgmask.apply(frame)
            kernel = np.ones((3, 3), np.uint8)
            fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
            frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
        return frame


cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

print(cv2.__version__)
hand = HandTrack()
bg = cv2.bgsegm.createBackgroundSubtractorMOG()
video_capture = cv2.VideoCapture(0)
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # face_detect(frame)
    frame = hand.remove_bg(frame)

    keypress = cv2.waitKey(1)
    if keypress & 0xFF == ord('q'):
        break
    if keypress & 0xFF == ord('b'):
        hand.get_background(frame)

    face_detect(frame)
    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
