import cv2
import numpy as np

def determine_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for contour in contours:
        # Calculate perimeter of contour
        peri = cv2.arcLength(contour, True)
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        shapes.append(approx)

    sign_shape = None
    for shape in shapes:
        if len(shape) == 3:
            sign_shape = "Triangle"
        elif len(shape) == 4:
            # Check if all angles are ~90 degrees
            (x, y, w, h) = cv2.boundingRect(shape)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                sign_shape = "Rectangle"
            else:
                sign_shape = "Rectangle"
        else:  
            _, _, w, h = cv2.boundingRect(approx)
            aspect_ratio = max(w, h) / min(w, h)
            if 0.8 <= aspect_ratio <= 1.2:
                sign_shape = "Circle"
            else:
                sign_shape = "None"

    return sign_shape