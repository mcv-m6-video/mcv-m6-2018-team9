import cv2
import numpy as np
import scipy.optimize as opt
import copy


class Tracker:

    def __init__(self, disappear_thr=3, min_matches=5, max_distance=100):
        self.filters = []
        self.min_matches = min_matches
        self.disappear_thr = disappear_thr
        self.obj_counter = 0
        self.max_distance = max_distance  # max distance traversed by an object
                                          # between frames

    def estimate(self, bboxes):
        # Centroid assignment through Hungarian algorithm
        blob_centroids = centroids(bboxes)
        kalman_centroids = [kf['last_prediction'][:2] for kf in self.filters]
        if kalman_centroids and blob_centroids.size != 0:
            kalman_centroids = np.array(kalman_centroids)
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
                unmatched_kalman[i] = None
                unmatched_blobs[j] = None
                # First correct, then predict
                self.filters[i]['kalman'].correct(blob_centroids[j])
                estimation = self.filters[i]['kalman'].predict().squeeze()
                self.filters[i]['last_prediction'] = estimation
                self.filters[i]['last_bbox'] = bboxes[j]
                self.filters[i]['match_count'] += 1  # increase counter
                self.filters[i]['disapp_count'] = 0  # reset counter

                if self.filters[i]['match_count'] >= self.min_matches:
                    result = dict(id=self.filters[i]['id'],
                                  location=estimation[:2],
                                  motion=estimation[2:],
                                  width=self.filters[i]['last_bbox'][2],
                                  height=self.filters[i]['last_bbox'][3])
                    result_list.append(result)

        unmatched_kalman = list(filter(lambda a: a is not None, unmatched_kalman))
        unmatched_blobs = list(filter(lambda a: a is not None, unmatched_blobs))

        # Manage unmatched kalman filters
        for i in reversed(unmatched_kalman):
            # Increase disappeared counter
            self.filters[i]['disapp_count'] += 1

            # Remove if exceeds threshold, otherwise predict
            if self.filters[i]['disapp_count'] < self.disappear_thr:
                estimation = self.filters[i]['kalman'].predict().squeeze()
                self.filters[i]['last_prediction'] = estimation

                if self.filters[i]['match_count'] >= self.min_matches:
                    result = dict(id=self.filters[i]['id'],
                                  location=estimation[:2],
                                  motion=estimation[2:],
                                  width=self.filters[i]['last_bbox'][2],
                                  height=self.filters[i]['last_bbox'][3])
                    result_list.append(result)
            else:
                del self.filters[i]

        # Manage unmatched blobs
        for i in unmatched_blobs:
            # Register new kalman filter
            new_filter = self._create_kalman_filter()
            self.filters.append(new_filter)

            # Add prediction to result list
            new_filter['kalman'].predict()  # needed before first correction
            new_filter['kalman'].correct(blob_centroids[i])
            estimation = new_filter['kalman'].predict().squeeze()
            new_filter['last_prediction'] = estimation
            new_filter['last_bbox'] = bboxes[i]
            new_filter['match_count'] += 1

            if new_filter['match_count'] >= self.min_matches:
                result = dict(id=new_filter['id'],
                              location=estimation[:2],
                              motion=estimation[2:],
                              width=bboxes[i][2],
                              height=bboxes[i][3])
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
        filter_object = dict(id=self.obj_counter,
                             kalman=kalman,
                             match_count=0,
                             disapp_count=0,
                             last_prediction=None,
                             last_bbox=None,)
        return filter_object


############################################
# Auxiliary functions
############################################


def find_bboxes(im):
    """Given a binary image, find its blobs and return their bounding boxes"""
    __, contours, __ = cv2.findContours(im, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cont) for cont in contours]
    return bboxes


def centroids(bboxes):
    """Take a sequence of bounding boxes and return their centroids

    The bounding boxes are specified as (x, y, w, h) tuples. The corresponding
    centroids are returned in a numpy array with shape [N, 2], where
    N = len(bboxes).

    """
    bboxes = np.array(bboxes, dtype='float32')
    x = bboxes[:, 0] + bboxes[:, 2] / 2
    y = bboxes[:, 1] + bboxes[:, 3] / 2

    return np.array([x, y], dtype='float32').T


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

def detect_blobs(im,im_out,params):
    """

    :param im: input image to track
    :param im_out: image to display bounding boxes
    :param params: Blob detector params
    :return:  blob, im_disp, bbox
    """


    detector = cv2.SimpleBlobDetector_create(params)

    blob_im = im
    blob = detector.detect(blob_im)

    print(blob)

    # Show blobs
    im_disp = im_out
    for k in blob:
        p1 = (int(k.pt[0] - k.size / 2), int(k.pt[1] - k.size / 2))
        p2 = (int(k.pt[0] + k.size / 2), int(k.pt[1] + k.size / 2))
        cv2.rectangle(im_disp, p1, p2, (255, 0, 0), 2, 1)

    bbox = (blob[0].pt[0],blob[0].pt[1], k.size/2, k.size/2)

    return im_disp, bbox


class MultipleObjectsTracker:
    """Multiple-objects tracker
        This class implements an algorithm for tracking multiple objects in
        a video sequence.
        The algorithm combines a saliency map for object detection and
        mean-shift tracking for object tracking.
    """
    def __init__(self, min_area=400, min_shift2=5):
        """Constructor
            This method initializes the multiple-objects tracking algorithm.
            :param min_area: Minimum area for a proto-object contour to be
                             considered a real object
            :param min_shift2: Minimum distance for a proto-object to drift
                               from frame to frame ot be considered a real
                               object
        """
        self.object_roi = []
        self.object_box = []

        self.min_cnt_area = min_area
        self.min_shift2 = min_shift2

        # Setup the termination criteria, either 100 iteration or move by at
        # least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          100, 1)

    def advance_frame(self, frame, proto_objects_map):
        """Advances the algorithm by a single frame
            This method tracks all objects via the following steps:
             - adds all bounding boxes from saliency map as potential
               targets
             - finds bounding boxes from previous frame in current frame
               via mean-shift tracking
             - combines the two lists by removing duplicates
            certain targets are discarded:
             - targets that are too small
             - targets that don't move
            :param frame: New input RGB frame
            :param proto_objects_map: corresponding proto-objects map of the
                                      frame
            :returns: frame annotated with bounding boxes around all objects
                      that are being tracked
        """
        self.tracker = copy.deepcopy(frame)

        # build a list of all bounding boxes
        box_all = []

        # append to the list all bounding boxes found from the
        # current proto-objects map
        box_all = self._append_boxes_from_saliency(proto_objects_map, box_all)

        # find all bounding boxes extrapolated from last frame
        # via mean-shift tracking
        box_all = self._append_boxes_from_meanshift(frame, box_all)

        # only keep those that are both salient and in mean shift
        if len(self.object_roi) == 0:
            group_thresh = 0  # no previous frame: keep all form saliency
        else:
            group_thresh = 1  # previous frame + saliency
        box_grouped, _ = cv2.groupRectangles(box_all, group_thresh, 0.1)

        # update mean-shift bookkeeping for remaining boxes
        self._update_mean_shift_bookkeeping(frame, box_grouped)

        # draw remaining boxes
        for (x, y, w, h) in box_grouped:
            cv2.rectangle(self.tracker, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        return self.tracker

    def _append_boxes_from_saliency(self, proto_objects_map, box_all):
        """Adds to the list all bounding boxes found with the saliency map
            A saliency map is used to find objects worth tracking in each
            frame. This information is combined with a mean-shift tracker
            to find objects of relevance that move, and to discard everything
            else.
            :param proto_objects_map: proto-objects map of the current frame
            :param box_all: append bounding boxes from saliency to this list
            :returns: new list of all collected bounding boxes
        """
        # find all bounding boxes in new saliency map
        box_sal = []
        _, cnt_sal, _ = cv2.findContours(proto_objects_map, 1, 2)
        for cnt in cnt_sal:
            # discard small contours
            if cv2.contourArea(cnt) < self.min_cnt_area:
                continue

            # otherwise add to list of boxes found from saliency map
            box = cv2.boundingRect(cnt)
            box_all.append(box)

        return box_all

    def _append_boxes_from_meanshift(self, frame, box_all):
        """Adds to the list all bounding boxes found with mean-shift tracking
            Mean-shift tracking is used to track objects from frame to frame.
            This information is combined with a saliency map to discard
            false-positives and focus only on relevant objects that move.
            :param frame: current RGB image frame
            :box_all: append bounding boxes from tracking to this list
            :returns: new list of all collected bounding boxes
        """
        print(box_all)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for i in range(len(self.object_roi)):
            roi_hist = copy.deepcopy(self.object_roi[i])
            box_old = copy.deepcopy(self.object_box[i])

            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, box_new = cv2.meanShift(dst, tuple(box_old), self.term_crit)
            self.object_box[i] = copy.deepcopy(box_new)

            # discard boxes that don't move
            (xo, yo, wo, ho) = box_old
            (xn, yn, wn, hn) = box_new

            co = [xo + wo/2, yo + ho/2]
            cn = [xn + wn/2, yn + hn/2]
            if (co[0]-cn[0])**2 + (co[1]-cn[1])**2 >= self.min_shift2:
                box_all.append(box_new)

        return box_all

    def _update_mean_shift_bookkeeping(self, frame, box_grouped):
        """Preprocess all valid bounding boxes for mean-shift tracking
            This method preprocesses all relevant bounding boxes (those that
            have been detected by both mean-shift tracking and saliency) for
            the next mean-shift step.
            :param frame: current RGB input frame
            :param box_grouped: list of bounding boxes
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        self.object_roi = []
        self.object_box = []
        for box in box_grouped:
            (x, y, w, h) = box
            hsv_roi = hsv[y:y + h, x:x + w]
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                               np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            self.object_roi.append(roi_hist)
            self.object_box.append(box)
