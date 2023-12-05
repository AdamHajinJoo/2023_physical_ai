import cv2
import numpy as np

def make_gray(img):
    """
    Convert an image to grayscale.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Grayscale image.
    """
    if len(img.shape) == 3:  # Color image
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # Grayscale
        return img

def adjust_brightness(img, darkening_factor):
    """
    Adjust the brightness of an image.

    Args:
        img (numpy.ndarray): Input image.
        darkening_factor (float): Factor to adjust brightness.

    Returns:
        numpy.ndarray: Adjusted image.
    """
    gray = make_gray(img)
    equ = cv2.equalizeHist(gray)
    adjusted_img = cv2.addWeighted(equ, 1 - darkening_factor, gray, darkening_factor, 0)
    return cv2.cvtColor(adjusted_img, cv2.COLOR_GRAY2BGR)

def gaussian_blur(img, blur_val):
    """
    Apply Gaussian blur to an image.

    Args:
        img (numpy.ndarray): Input image.
        blur_val (int): Blur kernel size.

    Returns:
        numpy.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(img, (blur_val, blur_val), 0)

def canny_edge(img):
    """
    Apply Canny edge detection to an image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Edges detected image.
    """
    return cv2.Canny(img, 50, 150, apertureSize=3)

def img_preprocess(img, blur_val, darkening_factor):
    """
    Preprocess an image by applying grayscale, brightness adjustment, Gaussian blur, and Canny edge detection.

    Args:
        img (numpy.ndarray): Input image.
        blur_val (int): Blur kernel size.
        darkening_factor (float): Factor to adjust brightness.

    Returns:
        numpy.ndarray: Processed image with edges detected.
    """
    gray = make_gray(img)
    ad_frame = adjust_brightness(gray, darkening_factor)
    blur = gaussian_blur(ad_frame, blur_val)
    edges = canny_edge(blur)
    return edges