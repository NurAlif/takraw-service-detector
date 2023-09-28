import numpy as np

# Function to calculate AP for a single class
def calculate_ap_for_class(ground_truth, predictions, iou_threshold=0.5):
    # Sort predictions by confidence score in descending order
    # predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

    true_positives = np.zeros(len(predictions))
    false_positives = np.zeros(len(predictions))
    total_gt_boxes = len(ground_truth)
    # print(predictions)

    if total_gt_boxes == 0:
        return 0.0  # No ground truth boxes, AP is 0

    for i, pred in enumerate(predictions):
        x1, y1, x2, y2, class_id, confidence = pred
        box = (x1, y1, x2, y2)

        best_iou = 0
        best_gt_idx = -1

        for j, gt_box in enumerate(ground_truth):
            # print("cls " + str(class_id) + " " + str(gt_box[4]))
            if gt_box[4] == class_id:
                iou = calculate_iou(box, gt_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives[i] = 1
            ground_truth[best_gt_idx][4] = -1  # Mark ground truth box as used
        else:
            false_positives[i] = 1

    # Calculate precision and recall


    cum_true_positives = np.cumsum(true_positives)
    cum_false_positives = np.cumsum(false_positives)
    
    recall = cum_true_positives / total_gt_boxes
    precision = cum_true_positives / (cum_true_positives + cum_false_positives)

    # Check if precision and recall arrays are not empty
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    # Calculate AP by interpolating precision-recall curve
    interpolated_precision = []
    for r in np.linspace(0, 1, 101):
        valid_precision_values = precision[recall >= r]
        if len(valid_precision_values) > 0:
            max_precision_at_recall_r = np.max(valid_precision_values)
            interpolated_precision.append(max_precision_at_recall_r)
    
    if len(interpolated_precision) == 0:
        return 0.0

    ap = np.mean(interpolated_precision)
    return ap

# Function to calculate Intersection over Union (IoU) between two boxes
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    intersection_x1 = max(x1, x1_gt)
    intersection_y1 = max(y1, y1_gt)
    intersection_x2 = min(x2, x2_gt)
    intersection_y2 = min(y2, y2_gt)

    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

# Function to calculate mAP for all classes and average them
def calculate_mAP_for_all_classes(ground_truth, predictions, num_classes, iou_threshold=0.5):
    ap_per_class = []
    for class_id in range(num_classes):
        class_ground_truth = [box for box in ground_truth if box[4] == class_id]
        class_predictions = [pred for pred in predictions if pred[4] == class_id]

        # print(class_predictions)
        ap = calculate_ap_for_class(class_ground_truth, class_predictions, iou_threshold)
        ap_per_class.append(ap)

    mean_ap = np.mean(ap_per_class)
    return mean_ap

