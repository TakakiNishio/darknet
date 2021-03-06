from ctypes import *
import math
import random
import cv2
import colorsys
import numpy as np
import argparse
import os
import copy

from mcnn import *
from network_structure import *


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    return (ctype * len(values))(*values)


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)



def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num = num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

# new
def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res



def detect_im(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = make_boxes(net)
    probs = make_probs(net)
    num = num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res


def detect_np(net, meta, np_img, thresh=.5, hier_thresh=.5, nms=.45):
    im = nparray_to_image(np_img)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def check_sign(rec):

    # print rec

    i = 0
    for x in rec:
        if x < 0:
            # print "negative value!"
            # print rec[i]
            rec[i] = 0
        i += 1

    # print rec

    return rec


#preparing to save the video
def initWriter(w, h, fps, save_path):
    fourcc = cv2.VideoWriter_fourcc('F','L','V','1')
    rec = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    return rec


if __name__ == "__main__":

    # setup some arguments
    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    parser.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    parser.add_argument('--label', '-l', type=str, default="carrot",help='specified label')
    parser.add_argument('--margin', '-m', type=int, default=15,help='margin value')
    parser.add_argument('--square', '-square', action="store_true")
    parser.add_argument('--save_name', '-s', type=str, default=False,help='camera ID')
    parser.add_argument('--superposition', '-superposition', action="store_true")
    args = parser.parse_args()

    # prepare to get image frames
    if not args.video_file == False:
        cap = cv2.VideoCapture(args.video_file)
    else:
        cap = cv2.VideoCapture(args.camera_ID)

    ret, img = cap.read()

    # Load YOLO
    net_filename = "../cfg/yolov3.cfg"
    weights_filename = "../yolov3.weights"
    meta_filename = "../cfg/coco.data"

    net = load_net(net_filename.encode('utf-8'), weights_filename.encode('utf-8'), 0)
    meta = load_meta(meta_filename.encode('utf-8'))
    
    # Load module type CNN model and weights
    model_path = 'carrot_model11/'
    model = MCNN32()
    image_size = 150
    mcnn = MCNN(model_path, model, image_size)

    color = (0,255,0)

    # prepare to record video
    ret, frame = cap.read()
    frame_height, frame_width, channels = frame.shape
    print("input size: (" + str(frame_height) + ", " + str(frame_width) + ")")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("input fps: " + str(fps))

    rec = False
    if not args.save_name == False:
        save_path = 'results/videos/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fps = 30
        rec = initWriter(frame_width, frame_height, fps, save_path+args.save_name)

    CNN_message = 'Searching with YOLOv3...'
    CNN_message_color = (255, 0, 255)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    m = args.margin
    im_cnt = 0

    while(1):
        ret, img = cap.read()
        copied_img = copy.deepcopy(img)

        if img is None:
            print("No image !!")
            break

        r = detect_np(net, meta, img)
        for i in r:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)

            if i[0].decode() == args.label:

                # print args.label + " is found!"
                # print (ymin, ymax, xmin, xmax)
                # print

                if args.square:
                    if w < h :
                        d = int((h - w)/2.0)
                        xmin = xmin - d
                        xmax = xmax + d
                    else:
                        d = int((w - h)/2.0)
                        ymin = ymin - d
                        ymax = ymax + d
                    rec_points = check_sign([ymin-m, ymax+m, xmin-m, xmax+m])
                    target_img = copied_img[rec_points[0]:rec_points[1], \
                                    rec_points[2]:rec_points[3]] # ymin:ymax, xmin:xmax

                    target_class, prob = mcnn(target_img)
                    

                else:
                    rec_points = check_sign([ymin-m, ymax+m, xmin-m, xmax+m])
                    target_img = copied_img[rec_points[0]:rec_points[1], \
                                        rec_points[2]:rec_points[3]]
                    # ymin:ymax, xmin:xmax

                    w, h, ch = target_img.shape[:3]

                    if args.superposition:
                        if w < h :
                            s = h
                            x1 = int((h-w)/2.0)
                            x2 = x1 + w
                            y1 = 0
                            y2 = h
                        else:
                            s = w
                            x1 = 0
                            x2 = w
                            y1 = int((w-h)/2.0)
                            y2 = y1 + h

                        background_img = np.tile(np.uint8([0,0,0]), (s,s,1))

                        background_img[x1:x2,y1:y2] = target_img

                        target_img = copy.deepcopy(background_img)
                        print("sup")

                        # cv2.imshow("aaa", background_img)

                        # target_class, prob = mcnn(target_img)
                        target_class, prob = mcnn(background_img)

                    else:
                        target_class, prob = mcnn(target_img)

                    
                if target_class == 0:
                    color = (0,0,255)
                    result = "(group: A)"
                    prob = 1.0 - prob
                else:
                    color = (255,0,0)
                    result = "(group: B)"

                # if target_class == 1 and prob > 0.95:
                #     color = (255,0,0)
                #     result = "(group: B)"
                # else:
                #     color = (0,0,255)
                #     result = "(group: A)"

                # if target_class == 1 and prob > 0.95:
                #     color = (255,0,0)
                #     result = "(label 1)"
                # elif target_class == 0 and (1.0-prob) > 0.95:
                #     color = (0,0,255)
                #     result = "(label 0)"
                # else:
                #     color = (0,255,0)
                #     result = "(else)"

                result_info = result + '(%2d%%)' % (prob*100)
 
                cv2.rectangle(img, pt1, pt2, color, 3, 1)

                # cv2.putText(img, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "%]",
                #             (pt1[0]+2, pt1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                # cv2.putText(img, i[0].decode() +" "+result+ " [" + str(round(i[1] * 100, 2)) + "%]",
                #             (pt1[0]+2, pt1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                # cv2.putText(img, i[0].decode() +" "+result,
                #             (pt1[0], pt2[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.putText(img, result_info,
                            (pt1[0], pt2[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                im_cnt += 1

        cv2.putText(img, CNN_message, (10,32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, CNN_message_color, 4)

        cv2.imshow("result", img)
        k = cv2.waitKey(1) & 0xFF

        if not args.save_name == False:
            rec.write(img)

        if k == 27:
            cv2.destroyAllWindows()
            exit()

    cap.release()
    if not args.save_name == False:
        rec.release()

