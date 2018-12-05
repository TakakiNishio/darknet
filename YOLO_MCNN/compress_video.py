import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from ctypes import *
import math
import random
import cv2
import colorsys
import numpy as np
import argparse
import os
import copy


#preparing to save the video
def initWriter(w, h, fps, save_path):
    fourcc = cv2.VideoWriter_fourcc('F','L','V','1')
    rec = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    return rec


# counting FPS
class fpsWithTick(object):
    def __init__(self):
        self._count     = 0
        self._oldCount  = 0
        self._freq      = 1000 / cv2.getTickFrequency()
        self._startTime = cv2.getTickCount()
    def get(self):
        nowTime         = cv2.getTickCount()
        diffTime        = (nowTime - self._startTime) * self._freq
        self._startTime = nowTime
        fps             = (self._count - self._oldCount) / (diffTime / 1000.0)
        self._oldCount  = self._count
        self._count     += 1
        fpsRounded      = round(fps, 1)
        return fpsRounded


if __name__ == "__main__":

    # setup some arguments
    parser = argparse.ArgumentParser(description='video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--save_name', '-s', type=str, default=False,help='filename')
    args = parser.parse_args()

    # prepare to get image frames
    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    # prepare to record video
    ret, frame = cap.read()
    frame_height, frame_width, channels = frame.shape
    print("input size: (" + str(frame_height) + ", " + str(frame_width) + ")")
    size = (int(frame_width/2), int(frame_height/2))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("input fps: " + str(fps))
    # fps = 30

    rec = False
    if not args.save_name == False:
        save_path = 'results/videos/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        rec = initWriter(frame_width, frame_height, fps, save_path+args.save_name)

    # main loop
    while(1):
        ret, img = cap.read()
        print("image !!")

        if img is None:
            print("No image !!")
            break

        result_img = cv2.resize(img,size)

        cv2.imshow("resized", result_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            exit()

        if not args.save_name == False:
            rec.write(img)
            # rec.write(result_img)            

    cap.release()
    if not args.save_name == False:
        rec.release()
