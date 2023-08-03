import cv2
import numpy as np
from numpy import random
import base64
from collections import deque


data_deque = {}

#This is the line from which the trails
line = [(100, 500), (1050, 500)]

"""
This draw_box() is used when we have to see the labels
and also Plots one bounding box on image img
"""
# def draw_box(img, box, label=None, color=None, line_thickness=None):
#     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

#         img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

#         cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

"""
This draw_box is used when we dont want label to be displayed
and also Plots one bounding box on image img
"""
def draw_box(img, box, label=None,color=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

"""
This draw_trails is used to Draws trails for tracked objects
using data deque
"""
def draw_trails(img, data_deque, color, thickness):
    for id, data in data_deque.items():
        for i in range(1, len(data)):
            if data[i - 1] is None or data[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, data[i - 1], data[i], color, thickness)

"""
This draw_counts function is used to Draws the counts for
 objects entering and leaving
"""
# def draw_counts(img, object_counter, object_counter1, width):
#     for idx, (key, value) in enumerate(object_counter1.items()):
#         cnt_str = str(key) + ":" + str(value)
#         cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
#         cv2.putText(img, 'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
#         cv2.line(img, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
#         cv2.putText(img, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

#     for idx, (key, value) in enumerate(object_counter.items()):
#         cnt_str1 = str(key) + ":" + str(value)
#         cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
#         cv2.putText(img, 'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
#         cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
#         cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

"""
This plot_boxes function is to to Plots bounding boxes
on image img, draws trails, and displays counts
"""
def plot_boxes(object_counter,object_counter1,img, bbox, names, object_id, identities=None, offset=(0, 0), line=None, line_thickness=None):
    cv2.line(img, line[0], line[1], (46, 162, 112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1
        draw_box(img, box, label=label, color=color, line_thickness=2)

    draw_trails(img, data_deque, color=[255, 0, 0], thickness=1)
    #draw_counts(img, object_counter, object_counter1, width)

"""
This compute_color_for_labels() function is used to give
unique color to different labels
"""
def compute_color_for_labels(label):
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = (0, 0, 0)
    return tuple(color)

"""
This draw_border() is used to Draws a border 
around the bounding box label
"""
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img

"""
This intersect() checks if two line segments intersect
"""
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
"""
This ccw() Checks if three points are in counter-clockwise order
"""
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

"""
This get_direction() calculates the direction based on two points
"""
def get_direction(point1, point2):
    direction_str = ""
    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

"""
This opencv_to_base64() is used to convert the 
given image into base64 format so that it can be presented on html
"""
def opencv_to_base64(image):
    retval, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode()
    return encoded_image


"""" 
Calculates the relative bounding box from absolute pixel values
"""
def xyxy_to_xywh(*xyxy):
    
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

