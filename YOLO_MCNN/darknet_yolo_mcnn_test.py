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


if __name__ == "__main__":

    # setup some arguments
    parser = argparse.ArgumentParser(description='yolov2_darknet_predict for video')
    parser.add_argument('--label', '-l', type=str, default="carrot",help='specified label')
    parser.add_argument('--margin', '-m', type=int, default=15,help='margin value')
    parser.add_argument('--square', '-square', action="store_true")
    parser.add_argument('--gray', '-gray', action="store_true")
    parser.add_argument('--superposition', '-superposition', action="store_true")
    parser.add_argument('--txt_directory', '-d', type=str, default="test_data/mixed",help='txt data path')
    parser.add_argument('--savedir', '-savedir', type=str, default=None,help='specified label')
    args = parser.parse_args()

    dataset_path = args.txt_directory
    end_flag = False
    fontType = cv2.FONT_HERSHEY_SIMPLEX

    image_path_txt = open(dataset_path+'/images.txt','r')
    label_txt = open(dataset_path+'/labels.txt','r')

    image_path_list = []
    label_list = []

    for image_path in image_path_txt:
        image_path_list.append(image_path.split('\n')[0])
        
    for label in label_txt:
        label_list.append(label.split('\n')[0])

    data_N = len(image_path_list)

    if args.savedir is not None:
        if not os.path.isdir(args.savedir):
            os.makedirs(args.savedir)

    # Load YOLO
    net = load_net("../cfg/yolov3.cfg", "../yolov3.weights", 0)
    meta = load_meta("../cfg/coco.data")

    # Load module type CNN model and weights
    model_path = 'carrot_model11/'
    model = MCNN32()
    image_size = 150

    # model_path = 'carrot_model32_black/'
    # model = MCNN32()
    # image_size = 150

    # model_path = 'carrot_model32_square/'
    # model = MCNN32()
    # image_size = 150

    # model_path = 'carrot_model_black_final/'
    # model = MCNN32()
    # image_size = 150

    # model_path = 'carrot_model_square_final/'
    # model = MCNN32()
    # image_size = 150

    # model_path = 'carrot_model_square_final_2/'
    # model = MCNN32()
    # image_size = 150

    # model_path = 'carrot_model_black_final_2/'
    # model = MCNN32()
    # image_size = 150


    if args.gray:
        mcnn = MCNN_grayed(model_path, model, image_size)
    else:
        mcnn = MCNN(model_path, model, image_size)

    color = (0,255,0)
    m = args.margin
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    cnt_success = 0

    # data_N = 10

    for data_index in range(data_N):

        img = cv2.imread('../'+image_path_list[data_index])
        correct_label = label_list[data_index]

        copied_img = copy.deepcopy(img)
        height = img.shape[0]
        width= img.shape[0]

        if img is None:
            print "No image !!"
            break

        r = detect_np(net, meta, img)

        cnt_negative = 0
        cnt_positive = 0

        for i in r:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)

            if i[0].decode() == args.label:

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
                    print prob

                else:
                    rec_points = check_sign([ymin-m, ymax+m, xmin-m, xmax+m])
                    target_img = copied_img[rec_points[0]:rec_points[1], \
                                        rec_points[2]:rec_points[3]]
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
                        target_class, prob = mcnn(target_img)
                        print prob
                    else:
                        target_class, prob = mcnn(target_img)
                        print prob
                    
                if target_class == 0:
                    color = (0,0,255)
                    result = "(group: A)"
                    cnt_negative += 1
                else:
                    color = (255,0,0)
                    result = "(group: B)"
                    cnt_positive += 1

                cv2.rectangle(img, pt1, pt2, color, 2)
                cv2.putText(img, i[0].decode() +" "+result,
                            (pt1[0]+2, pt1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        output_label = str(cnt_negative) + str(cnt_positive)

        if output_label == correct_label:
            cnt_success += 1.0
            msg_result = "success"
            color_msg = (200,0,0)
        else:
            msg_result = "failure"
            color_msg = (0,0,200)

        msg_num = "image: " + str(data_index+1) + "/" + str(data_N)
        msg_correct_label = "correct label: " + correct_label
        msg_output_label = "output label: " + output_label
            
        print
        print msg_num
        print msg_correct_label
        print msg_output_label
        print msg_result

        # cv2.putText(img, msg_num, (5, height-20), fontType, 0.7, (255,0,0), 2)
        # cv2.putText(img, msg_correct_label, (5,40), fontType, 0.9, (0,255,0), 2)
        # cv2.putText(img, msg_output_label, (5,75), fontType, 0.9, (255,0,0), 2)
        # cv2.putText(img, msg_result, (5,105), fontType, 0.9, color_msg, 2)

        cv2.imshow("result", img)

        if args.savedir is not None:
            cv2.imwrite(args.savedir+'/'+str(data_index)+'_'+msg_result+'.jpg', img)

        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            exit()

    success_rate = (cnt_success/data_N)*100
    print
    print "test accuracy: " + str(round(success_rate,2))

    acc_img = np.tile(np.uint8([245,245,245]), (height, width,1))

    cv2.putText(acc_img, "accuracy: "+str(round(success_rate,2))+" %",
                (25,150), fontType, 0.8, (255,0,0), 2)

    cv2.imshow("result", acc_img)
    key = cv2.waitKey(1500) & 0xFF

    cv2.destroyAllWindows()
            
