import os
import cv2
import numpy as np
from iyolo import get_perf, get_yolo, CLASSES
from yolo.calcap import calculate_ap_for_class, calculate_iou, calculate_mAP_for_all_classes

def read_yolo_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        x1 = (x_center - width / 2)
        y1 = (y_center - height / 2)
        x2 = (x_center + width / 2)
        y2 = (y_center + height / 2)
        boxes.append([x1, y1, x2, y2, class_id])
    return np.array(boxes)

def calculate_mAP(dataset_folder, iou_threshold=0.5):
    # Iterate through all images in the dataset
    # For each image, call infer to get predictions
    # Compare predictions to ground truth labels and calculate AP for each class
    # Compute mean AP (mAP) over all classes and images
    mean_ap = 0.0
    total_images = 0

    for image_filename in os.listdir(os.path.join(dataset_folder, 'images')):
        image_path = os.path.join(dataset_folder, 'images', image_filename)
        label_path = os.path.join(dataset_folder, 'labels', image_filename.replace('.jpg', '.txt'))

        if os.path.exists(label_path):
            total_images += 1
            image = cv2.imread(image_path)
            predicted_boxes, draw = YOLO.infer(image)
            ground_truth_boxes = read_yolo_labels(label_path)

            ground_truth = [[x1, y1, x2, y2, int(class_id)] for x1, y1, x2, y2, class_id in ground_truth_boxes]
            # print(ground_truth)
            predictions = [[x1/480, y1/320, x2/480, y2/320, class_id, confidence] for x1, y1, x2, y2, class_id, confidence in predicted_boxes]
            # print(predictions)
            num_classes = int(max([box[4] for box in ground_truth])) + 1
 

            iou_threshold = 0.5
            mAP = calculate_mAP_for_all_classes(ground_truth, predictions, num_classes, iou_threshold)

            mean_ap+=mAP
    print("selesai")

    # Calculate mean AP
    if total_images > 0:
        mean_ap /= total_images
    return mean_ap

YOLO = get_yolo("v7")

dataset_folder = "/media/name/praptana/processed/splited/test"
mAP = calculate_mAP(dataset_folder, iou_threshold=0.5)
print(f"mAP at IoU 0.5: {mAP}")