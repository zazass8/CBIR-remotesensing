import cv2
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
import numpy as np

def calculate_color_histogram(image, bins=64):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the 2D histogram
    hist = cv2.calcHist([hsv_image], [0, 1], None, [bins, bins], [0, 180, 0, 256])

    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def calculate_lbp_texture(image, num_points=24, radius=8):
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

def calculate_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    return np.log(np.abs(hu_moments))


def calculate_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # Convert the image to grayscale if it's not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate GLCM
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Calculate GLCM properties (you can choose which properties to use)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    # Return the selected GLCM features
    return contrast, dissimilarity, homogeneity, energy, correlation