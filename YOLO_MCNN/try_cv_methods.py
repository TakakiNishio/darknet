#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import argparse


if __name__ == '__main__':

    # setup some arguments
    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    args = parser.parse_args()

    # prepare to get image frames
    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    # Boosting
    # tracker = cv2.TrackerBoosting_create()

    # MIL
    # tracker = cv2.TrackerMIL_create()

    # KCF
    tracker = cv2.TrackerKCF_create()

    # TLD
    # tracker = cv2.TrackerTLD_create()

    # MedianFlow
    # tracker = cv2.TrackerMedianFlow_create()

    # GOTURN
    # https://github.com/opencv/opencv_contrib/issues/941#issuecomment-343384500
    # https://github.com/Auron-X/GOTURN-Example
    # http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
    # tracker = cv2.TrackerGOTURN_create()


    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        key = cv2.waitKey(20) & 0xFF

        cv2.imshow("frame", frame)

        if key == ord("a"):
            bbox = (0,0,10,10)
            bbox = cv2.selectROI(frame, False)
            ok = tracker.init(frame, bbox)
            cv2.destroyAllWindows()
            break
        if key == ord("q"):
            print("quit")
            break


    while True:
        # VideoCapture
        ret, frame = cap.read()

        if not ret:
            continue

        # Start timer
        timer = cv2.getTickCount()
        track, bbox = tracker.update(frame)

        # FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if track:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else :
            cv2.putText(frame, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        # FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA);

        cv2.imshow("frame", frame)

        k = cv2.waitKey(1)
        if k == 27 :
            break

    cap.release()
    cv2.destroyAllWindows()
