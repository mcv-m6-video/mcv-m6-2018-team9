import cv2
import numpy as np


class Tracker:

    def __init__(self, disappear_threshold):
        self.filters = []
        self.dapp_thr = disappear_threshold
        self.obj_counter = 0

    def estimate(self, bboxes):
        centroids = self._bboxes2centroids_(bboxes)
        self._assign_objects_(centroids)
        predictions = []
        for f in self.filters:
            if f['bbox_id'] is not None:
                f['kalman'].correct(centroids[f['bbox_id']])
                pred = f['kalman'].predict()
                predictions.append(pred)
        return predictions

    def _assign_objects_(self, centroids):
        pass

    def _create_kalman_filter_(self):
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

    def _bboxes2centroids_(self, bboxes):
        centroids = []
        for bb in bboxes:
            centroids.append([(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2])
        return np.array(centroids)
