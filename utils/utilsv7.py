import cv2

import numpy as np

import os
import pickle
from collections import defaultdict, namedtuple, OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import onnx
import tensorrt as trt
import torch

CLASSES = ['ball', 'gpost', 'robot']
COLORS = [(200,40,40),(40,200,40),(40,40,200)]

def postprocess_v7(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes

def init_tensorrt_v7(engine, device):

    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_tensor_name(index)
        dtype = trt.nptype(model.get_tensor_dtype(name))
        shape = tuple(model.get_tensor_shape(name))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

        # warmup for 10 times
    for _ in range(10):
        tmp = torch.randn(1,3,480,320).to(device)
        binding_addrs['images'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))

    return bindings, binding_addrs, context

def letterbox_v7(im, new_shape=(480, 320), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad: 
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)