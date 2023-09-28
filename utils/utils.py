import cv2
def draw_border(canvas, color):
    return cv2.rectangle(canvas, (2,2), (638, 478), color, 3)