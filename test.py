#!/usr/bin/env python3

import cv2 
import torch
import time
import numpy as np

import math
import time
import streamer as streamer

from iyolo import get_yolo, get_perf, CLASSES


CLASSES = ['ball', 'gpost', 'robot']
COLORS = [(200,40,40), (40,200,40), (40,40,200)]

YOLO = None

videocap = None

def detect():
    bgr = videocap.read_in()

    dets, draw = YOLO.infer(bgr)

    videocap.store_out(draw)

def startInference():
    global YOLO

    global videocap

    YOLO = get_yolo("nas")
    
    videocap = streamer.VideoCapture()
    track = streamer.VideoOpencvTrack(videocap)
    streamer.video = track

    print('inference is running...')
    return 0

def inferenceLoop():
    count = 0
    while(count < 10000):
        detect()
        count+=1
    infer_only, pipe = get_perf()

    print(infer_only)
    print(pipe)

def shutdown():
    videocap.release()


startInference()
inferenceLoop()
