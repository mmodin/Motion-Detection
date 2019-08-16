import cv2
import time
import imutils
import numpy as np
from datetime import datetime


class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def release(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        return frame

    def show_frames(self):
        success, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)


class FrameDelta:
    def __init__(self, min_area=500, max_age=10):
        self.reference_frame = None
        self.min_area = min_area
        self.text = "No motion"
        self.ref_frame_creation_time = time.time()
        self.ref_frame_age = 0
        self.ref_frame_max_age = max_age

    def validate_ref_frame_age(self):
        self.ref_frame_age = time.time() - self.ref_frame_creation_time
        if self.ref_frame_age > self.ref_frame_max_age:
            print("Resetting reference frame after {} seconds".format(
                self.ref_frame_age))
            self.reference_frame = None
            self.ref_frame_creation_time = time.time()

    def calculate_and_show(self, frame):
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.reference_frame is None:
            self.reference_frame = gray
        frame_delta = cv2.absdiff(self.reference_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                self.text = "No motion"
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.text = "MOTION DETECTED!"
        cv2.putText(frame, "Room status: {}".format(self.text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frame_delta)
        self.validate_ref_frame_age()


class FPS:
    def __init__(self):
        self.total_frame_count = 0
        self.start_time = None
        self.frames = None

    def start_counting(self):
        self.start_time = time.time()
        self.frames = 0

    def count(self):
        self.frames += 1
        self.total_frame_count += 1
        time_since_start = time.time() - self.start_time
        if time_since_start > 1:
            print(self.frames)
            self.start_counting()


if __name__ == "__main__":
    cam = Camera()
    fps = FPS()
    fd = FrameDelta()
    frame_count = 0
    fps.start_counting()
    while True:
        fd.calculate_and_show(cam.get_frame())
        fps.count()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
