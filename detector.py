import argparse
import time
import glob, os
import numpy as np
import cv2

from PIL import Image
from PIL import ImageDraw

import nms

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]

MODEL = "models/best_exp_e100-int8_edgetpu.tflite"

CONF_TH = 0.5

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

def resize_and_pad(image, desired_size):
    old_size = image.shape[:2] 
    ratio = float(desired_size/max(old_size))
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # new_size should be in (width, height) format
    
    image = cv2.resize(image, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    
    pad = (delta_w, delta_h)
    
    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,
        value=color)
        
    return new_im, pad
     
def get_image_tensor(img, max_size, debug=False):
    """
    Reshapes an input image into a square with sides max_size
    """
    if type(img) is str:
        img = cv2.imread(img)
    
    resized, pad = resize_and_pad(img, max_size)
    resized = resized.astype(np.float32)
    
    if debug:
        cv2.imwrite("/home/valdis/work/nets_validation/intermediate.png", resized)

    # Normalise!
    resized /= 255.0
    
    return img, resized, pad

def get_scaled_coords(xyxy, pad, input_size, out_w, out_h):
    """
    Converts raw prediction bounding box to orginal
    image coordinates.
    
    Args:
        xyxy: array of boxes
        output_image: np array
        pad: padding due to image resizing (pad_w, pad_h)
    """
    pad_w, pad_h = pad
    in_h, in_w = input_size
    #out_h, out_w, _ = output_image.shape
            
    ratio_w = out_w/(in_w - pad_w)
    ratio_h = out_h/(in_h - pad_h) 
    
    out = []
    for coord in xyxy:

        x1, y1, x2, y2 = coord
                    
        x1 *= in_w*ratio_w
        x2 *= in_w*ratio_w
        y1 *= in_h*ratio_h
        y2 *= in_h*ratio_h
        
        x1 = max(0, x1)
        x2 = min(out_w, x2)
        
        y1 = max(0, y1)
        y2 = min(out_h, y2)
        
        out.append((x1, y1, x2, y2))
    
    return np.array(out).astype(int)


def Thread_detect(rgb_width, rgb_height, detect_framebuffer, detector_results):

    w = rgb_width
    h = rgb_height

    conf_thresh = 0.001
    iou_thresh = 0.6
    classes = None
    filter_classes = None #same as classes?
    agnostic_nms = False
    max_det = 100

    interpreter = make_interpreter(MODEL)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_zero = input_details[0]['quantization'][1]
    input_scale = input_details[0]['quantization'][0]
    output_zero = output_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    
    print("Successfully loaded {}".format(MODEL))

    input_size = common.input_size(interpreter)

    pad = (0, 140)#(delta_w, delta_h)
    pad_color = [100, 100, 100]

    time0 = time.time()
    while True:

        if not detect_framebuffer.empty():
            print("fps;", 1/(time.time() - time0))
            time0 = time.time()

            frame = detect_framebuffer.get()
        
            #resized, pad = resize_and_pad(frame, input_size[0])
            resized = cv2.copyMakeBorder(frame, 0, pad[1], 0, pad[0], cv2.BORDER_CONSTANT, value=pad_color)
            resized = resized.astype(np.float32)

            # Normalise!
            resized /= 255.0

            if net_image.shape[0] == 3:
                #print("Transpose")
                net_image = net_image.transpose((1,2,0))

            #net_image = net_image.astype('float32')

            net_image = (net_image/input_scale) + input_zero
            net_image = net_image[np.newaxis].astype(np.uint8)

            tstart = time.time()

            interpreter.set_tensor(input_details[0]['index'], net_image)
            interpreter.invoke()

            # Scale output
            result = (common.output_tensor(interpreter, 0).astype('float32') - output_zero) * output_scale
            inference_time = time.time() - tstart
            boxes = result[0,:,0:4]
            confs = result[0,:,4]
            scores = result[0,:,5:]

            #print(result.shape, boxes.shape, confs.shape)#[xywh, conf, class0, class1, ...]

            tstart = time.time()
            #nms_result = nms.non_max_suppression(result, conf_thresh, iou_thresh, filter_classes, agnostic_nms, max_det=max_det)
            nms_result = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, iou_thresh) 
            nms_time = time.time() - tstart

            bbox = get_scaled_coords(result[0,nms_result,0:4], pad, input_size, w ,h)#x1, y1, x2, y2
            conf = confs[nms_result]
            score = scores[nms_result]

            predictions = []
            num = len(nms_result)

            for n in range(num):#center x, centery, w,h
                #bboxS = (int(bbox[n,0] - bbox[n,2]/2), int(bbox[n,1] - bbox[n,3]/2), int(bbox[n,2]), int(bbox[n,3]))
                #cv2.rectangle(full_image, bboxS, (0,255,0), 1)

                obj_id = np.where(score[n] == max(score[n]))[0][0]

                #print("class", obj_id, conf[n])

                label = "person"#labels.get(obj_id, obj_id)
                #cv2.putText(full_image, label, (int(bbox[n,0] - bbox[n,2]/2), int(bbox[n,1] - bbox[n,3]/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))

                left = (bbox[n,0] - bbox[n,2]/2)/w
                top = (bbox[n,1] - bbox[n,3]/2)/h
                width = bbox[n,2]/w
                height = bbox[n,3]/h

                if float(conf[n]) > CONF_TH:#labels.get(obj.id, obj.id)
                    print(obj_id, label, left, top, width, height)
                    #detector_results.put((obj_id, label, left, top, width, height))