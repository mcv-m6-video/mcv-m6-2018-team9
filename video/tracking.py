import cv2
import numpy as np
import scipy.optimize as opt


class Tracker:

    def __init__(self, disappear_thr, max_distance=50):
        self.filters = []
        self.disappear_thr = disappear_thr
        self.obj_counter = 0
        self.max_distance = max_distance  # max distance traversed by an object
                                          # between frames

    def estimate(self, bboxes):
        # Centroid assignment through Hungarian algorithm
        blob_centroids = centroids(bboxes)
        kalman_centroids = [kf['last_prediction'][:2] for kf in self.filters]
        if kalman_centroids:
            kalman_centroids = np.vstack(kalman_centroids)
            dist = euclidean_distance(kalman_centroids, blob_centroids)
            match_i, match_j = opt.linear_sum_assignment(dist)
        else:
            match_i = []
            match_j = []

        result_list = []
        unmatched_kalman = list(range(len(self.filters)))
        unmatched_blobs = list(range(len(blob_centroids)))

        # Manage matched kalman filters
        for i, j in zip(match_i, match_j):
            if dist[i, j] <= self.max_distance:
                unmatched_kalman.remove(i)
                unmatched_blobs.remove(j)

                # FIXME: invoke predict() after correct()???
                self.filters[i]['kalman'].correct(blob_centroids[j])
                estimation = self.filters[i]['kalman'].predict()
                self.filters[i]['last_prediction'] = estimation
                result = dict(id=self.filters[i]['id'],
                              location=estimation[:2],
                              motion=estimation[2:])
                result_list.append(result)

                # Reset disappeared counter
                self.filters[i]['disappeared'] = 0

        # Manage unmatched kalman filters
        for i in reversed(unmatched_kalman):
            # Increase disappeared counter
            self.filters[i]['disappeared'] += 1

            # Remove if exceeds threshold, otherwise predict
            if self.filters[i]['disappeared'] < self.disappear_thr:
                estimation = self.filters[i]['kalman'].predict()
                self.filters[i]['last_prediction'] = estimation
                result = dict(id=self.filters[i]['id'],
                              location=estimation[:2],
                              motion=estimation[2:])
                result_list.append(result)
            else:
                del self.filters[i]

        # Manage unmatched blobs
        for i in unmatched_blobs:
            # Register new kalman filter
            new_filter = self._create_kalman_filter()
            self.filters.append(new_filter)

            # Add prediction to result list
            new_filter['kalman'].correct(blob_centroids[i])
            estimation = new_filter['kalman'].predict()
            new_filter['last_prediction'] = estimation
            result = dict(id=new_filter['id'],
                          location=estimation[:2],
                          motion=estimation[2:])
            result_list.append(result)

        return result_list

    def _create_kalman_filter(self):
        self.obj_counter += 1
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        filter_object = dict(id=self.obj_counter, kalman=kalman, disappeared=0)
        return filter_object


############################################
# Auxiliary functions
############################################


def centroids(bboxes):
    result = []
    for b in bboxes:
        result.append([(b[2] + b[0]) / 2, (b[3] + b[1]) / 2])
    return np.array(result, dtype='float32')


def euclidean_distance(a, b):
    """Compute the euclidean distance between points in a and b

    Args:
      a: numpy array with shape Nx2 containing N 2d points.
      b: numpy array with shape Mx2 containing M 2d points.

    Returns:
      A matrix with shape NxM with the euclidean distance of each 2D vector in
      'a' with respect to each vector in 'b'.

    """
    n = a.shape[0]
    m = b.shape[0]
    result = np.empty((n, m), dtype='float32')

    for i in range(n):
        result[i] = np.linalg.norm(b - a[i], axis=1)

    return result
