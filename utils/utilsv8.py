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

def postprocess_v8(data):
    assert len(data) == 4
    num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][
        0], data[3][0]
    nums = num_dets.item()
    if nums == 0:
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, ))
    # check score negative
    scores[scores < 0] = 1 + scores[scores < 0]

    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]

    return bboxes, scores, labels

def letterbox_v8(im, new_shape = (480, 320), color = (114, 114, 114)):
    
    shape = im.shape[:2] 
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    
    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad:  
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)
    return im, r, (dw, dh)

def blob_v8(im):
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32)

    return im

class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.uint8: torch.uint8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()
        num_bindings = model.num_bindings
        names = [model.get_binding_name(i) for i in range(num_bindings)]

        self.bindings: List[int] = [0] * num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_bindings = num_bindings
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.idx = list(range(self.num_outputs))

    def __init_bindings(self) -> None:
        idynamic = odynamic = False
        Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
        inp_info = []
        out_info = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                idynamic |= True
            inp_info.append(Tensor(name, dtype, shape))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                odynamic |= True
            out_info.append(Tensor(name, dtype, shape))

        if not odynamic:
            self.output_tensor = [
                torch.empty(info.shape, dtype=info.dtype, device=self.device)
                for info in out_info
            ]
        self.idynamic = idynamic
        self.odynamic = odynamic
        self.inp_info = inp_info
        self.out_info = out_info

    def set_profiler(self, profiler: Optional[trt.IProfiler]):
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def set_desired(self, desired: Optional[Union[List, Tuple]]):
        if isinstance(desired,
                      (list, tuple)) and len(desired) == self.num_outputs:
            self.idx = [self.output_names.index(i) for i in desired]

    def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:

        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[torch.Tensor] = [
            i.contiguous() for i in inputs
        ]

        for i in range(self.num_inputs):
            self.bindings[i] = contiguous_inputs[i].data_ptr()
            if self.idynamic:
                self.context.set_binding_shape(
                    i, tuple(contiguous_inputs[i].shape))

        outputs: List[torch.Tensor] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.odynamic:
                shape = tuple(self.context.get_binding_shape(j))
                output = torch.empty(size=shape,
                                     dtype=self.out_info[i].dtype,
                                     device=self.device)
            else:
                output = self.output_tensor[i]
            self.bindings[j] = output.data_ptr()
            outputs.append(output)

        self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        self.stream.synchronize()

        return tuple(outputs[i]
                     for i in self.idx) if len(outputs) > 1 else outputs[0]