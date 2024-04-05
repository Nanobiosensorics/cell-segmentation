import numpy as np
import cv2 as cv
from src.methods.method import Method, add_execution_time


class WatershedMethod(Method):
    @add_execution_time
    def predict(self, im_mic):
        gray = cv.cvtColor(np.asarray(im_mic), cv.COLOR_RGB2GRAY)

        # Thresholding the image.
        m_foreground = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 7)

        # Trying to differentiate foreground from background.
        h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
        v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 25))
        m_foreground = cv.morphologyEx(m_foreground, cv.MORPH_CLOSE, h_kernel, iterations=4)
        m_foreground = cv.morphologyEx(m_foreground, cv.MORPH_CLOSE, v_kernel, iterations=4)
        m_foreground = cv.morphologyEx(m_foreground, cv.MORPH_TOPHAT, h_kernel, iterations=5)
        m_foreground = cv.morphologyEx(m_foreground, cv.MORPH_TOPHAT, v_kernel, iterations=5)

        # Calculating the background sure of by dilating.
        sure_bg = cv.dilate(m_foreground, np.ones((5, 5)), iterations=4)

        # Calculating the sure foreground.
        dist_transform = cv.distanceTransform(m_foreground, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Calculating the unknown region.
        unknown = cv.subtract(sure_bg,sure_fg)

        # Applying the watershed algorithm.
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0

        markers = cv.watershed(np.asarray(im_mic), markers)
        markers[markers == -1] = 1
        markers[True] -= 1

        return markers

