import cv2
import numpy as np


class Tracker:

    def __init__(self, disappear_threshold):
        self.filters = []
        self.dapp_thr = disappear_threshold
        self.obj_counter = 0

    def estimate(self):
        pass

    def assign_objects(self):
        pass

    def create_kalman_filter(self):
        self.obj_counter += 1
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        filter_object = dict(id=self.obj_counter, kalman=kalman, disappeared=0,
                             bbox_id=0)
        return filter_object
