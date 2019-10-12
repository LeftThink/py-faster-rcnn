#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time 
import pdb 

font=cv2.FONT_HERSHEY_PLAIN
CLASSES = ('__background__', 'p')

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #im = im[:, :, (2, 1, 0)]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        x0,y0,x1,y1 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        cv2.rectangle(im, (x0,y0), (x1,y1), (0,255,0), 1, 0)
        cv2.putText(im, "{:.3f}".format(score), (x0,y0-10), font, 1.0, (0,0,255), thickness=1)

    #h,w,_ = im.shape 
    #im = cv2.resize(im, (int(w/2),int(h/2)))
    #cv2.imshow('preview', im) 
    #if (cv2.waitKey(-1) & 0xFF) == ord('q'):
    #    return

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--def', dest='proto', help='caffe prototxt file', type=str)
    parser.add_argument('--net', dest='model', help='caffe model', type=str)
    parser.add_argument('-i', '--input', dest='input', help='input data', type=str)
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = args.proto
    caffemodel = args.model 

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    print '\n\nLoaded network {:s}'.format(caffemodel)
    #pdb.set_trace()
    
    # Warmup on a dummy image
    input_data = args.input 
    if os.path.isdir(input_data):
        print("not supported!")
        sys.exit(0)

    _, ext = os.path.splitext(input_data)
    if ext in ['.txt']:
        pass
    elif ext in ['.jpg', '.jpeg']:
        pass 
    elif ext in ['.mp4', '.h264', '.avi']:
        vc = cv2.VideoCapture(input_data)
        if vc.isOpened():
            rt, frame = vc.read()
        else:
            raise Exception("Read Input Video Error")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ts = time.strftime("%m%d%H%M%S".format(time.localtime))
        outfile = "output.{:s}.{:s}".format(ts,os.path.basename(input_data))
        out = cv2.VideoWriter(outfile, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        while rt:
            demo(net, frame)
            out.write(frame)
            rt, frame = vc.read()
    else:
        print("not supported!")
        pass 

