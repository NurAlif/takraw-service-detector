import numpy as np
import math

def are_ellipses_colliding(bboxa, bboxb):
    bboxa = np.array(bboxa)
    bboxb = np.array(bboxb)

    center_a = (bboxa[:2] + bboxa[2:]) / 2
    center_b = (bboxb[:2] + bboxb[2:]) / 2

    semi_major_axis_a = abs(bboxa[2] - bboxa[0]) / 2
    semi_minor_axis_a = abs(bboxa[3] - bboxa[1]) / 2

    semi_major_axis_b = abs(bboxb[2] - bboxb[0]) / 2
    semi_minor_axis_b = abs(bboxb[3] - bboxb[1]) / 2

    distance = np.linalg.norm(center_b - center_a)

    sum_of_radii = semi_major_axis_a + semi_major_axis_b

    return distance <= sum_of_radii

def is_ellipse_inside(bboxa, bboxb): # is a inside b
    bboxa = np.array(bboxa)
    bboxb = np.array(bboxb)

    center_a = (bboxa[:2] + bboxa[2:]) / 2
    center_b = (bboxb[:2] + bboxb[2:]) / 2

    semi_major_axis_a = abs(bboxa[2] - bboxa[0]) / 2
    semi_minor_axis_a = abs(bboxa[3] - bboxa[1]) / 2

    semi_major_axis_b = abs(bboxb[2] - bboxb[0]) / 2
    semi_minor_axis_b = abs(bboxb[3] - bboxb[1]) / 2

    distance_x = abs(center_b[0] - center_a[0])
    distance_y = abs(center_b[1] - center_a[1])

    return (distance_x + semi_major_axis_a <= semi_major_axis_b) and (distance_y + semi_minor_axis_a <= semi_minor_axis_b)

def calculate_vector(bboxa, bboxb): # a to b
    bboxa = np.array(bboxa)
    bboxb = np.array(bboxb)

    center_a = (bboxa[:2] + bboxa[2:]) / 2
    center_b = (bboxb[:2] + bboxb[2:]) / 2

    vector = tuple(center_b - center_a)

    return vector

def calculate_angle(bboxa, bboxb): # a to b
    bboxa = np.array(bboxa)
    bboxb = np.array(bboxb)

    center_a = (bboxa[:2] + bboxa[2:]) / 2
    center_b = (bboxb[:2] + bboxb[2:]) / 2

    vector = tuple(center_b - center_a)

    return math.atan2(vector[1], vector[0])

def is_angle_difference_below_limit(angle1, angle2, limit):
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)
    
    angle_difference = abs(angle1 - angle2)
    
    return angle_difference < limit

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = area_bbox1 + area_bbox2 - intersection_area

    iou = intersection_area / union_area

    return iou

def is_iou_above_limit(bbox1, bbox2, limit):
    iou = calculate_iou(bbox1, bbox2)
    return iou > limit

def find_closest_bbox_with_class(dets, target_bbox, class_target):
    closest_bbox = None
    closest_distance = float('inf')

    for bbox in dets:
        x1, y1, x2, y2, class_index, _ = bbox

        if class_index == class_target:
            target_center_x = (target_bbox[0] + target_bbox[2]) / 2
            target_center_y = (target_bbox[1] + target_bbox[3]) / 2

            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2

            distance = np.sqrt((target_center_x - bbox_center_x)**2 + (target_center_y - bbox_center_y)**2)

            if distance < closest_distance:
                closest_bbox = bbox
                closest_distance = distance

    if closest_bbox == None: return target_bbox, False
    return closest_bbox[:4], True
