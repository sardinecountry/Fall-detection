import numpy as np
import cv2


def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 0, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return frame


def open_demo(gray):
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def get_diff(current_frame, last_frame):
    last_frame_binary = preprocess(last_frame)
    current_frame_binary = preprocess(current_frame)
    d1 = current_frame_binary.astype(np.float32) - last_frame_binary.astype(np.float32)
    d2 = last_frame_binary.astype(np.float32) - current_frame_binary.astype(np.float32)
    d = d1 - d2
    d = np.abs(d)
    d[d > 255] = 255
    d = d.astype(np.uint8)

    d_open = open_demo(d)
    return d_open


def get_box(image):
    row_sum = image.sum(axis=1)
    col_sum = image.sum(axis=0)

    indices = np.where(row_sum != 0)[0]
    if len(indices) == 0:
        return None
    row_min, row_max = indices[0], indices[-1]

    indices = np.where(col_sum != 0)[0]
    if len(indices) == 0:
        return None
    col_min, col_max = indices[0], indices[-1]

    x_min, x_max = col_min, col_max
    y_min, y_max = row_min, row_max
    return x_min, y_min, x_max, y_max
