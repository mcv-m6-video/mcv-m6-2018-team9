import numpy as np
import cv2
from data import cdnet
from evaluation import metrics, animations
from video import bg_subtraction, morphology, video_stabilization, tracking
import matplotlib.pyplot as plt
import time


def run(dataset):
    """Use kalman filter to generate a video sequence

    Args:
      dataset: (str) 'highway' or 'traffic'

    """
    # Dataset-specific parameters
    if dataset == 'highway':
        stabilization = False

        # background subtraction model
        rho = 0.20
        alpha = 2.8

        # morphology
        bsize = 50
        se_close = (15, 15)
        se_open = (4, 17)
        shadow_t1 = 0.082
        shadow_t2 = 0.017

    elif dataset == 'traffic':
        stabilization = False

        # background subtraction model
        rho = 0.15
        alpha = 2.25

        # morphology
        bsize = 400
        se_close = (15, 15)
        k = 9
        l = 30
        se_open = np.eye(l, dtype=np.uint8)

        for r in range(0, k):
            se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r + 1))
            se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r - 1))
        se_open = np.transpose(se_open.astype(np.uint8))

    # Read dataset
    train, gt_train = cdnet.read_sequence('week5', dataset, 'train',
                                          colorspace='gray',annotated=True)
    test, gt_test = cdnet.read_sequence('week5', dataset, 'test',
                                        colorspace='gray', annotated=True)

    if stabilization:
        # Stabilize sequences
        train, train_mask = video_stabilization.ngiaho_stabilization(
            train, gt_train, '.', 0, 320)
        test, test_mask = video_stabilization.ngiaho_stabilization(
            test, gt_test, '.', 0, 320)

        # Add axis
        train = train[..., np.newaxis]
        test = test[..., np.newaxis]

        # Adaptive model prediction
        model = bg_subtraction.create_model_mask(train, train_mask[1,:])
        pred = bg_subtraction.predict_masked(test, test_mask[1], model, alpha,
                                             rho=rho)
    else:
        # Adaptive model prediction
        model = bg_subtraction.create_model(train)
        pred = bg_subtraction.predict(test, model, alpha, rho=rho)

    # Imfill + filter small blobs
    filled8 = morphology.imfill(pred, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)

    # Closing
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_close)
    clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE, st_elem)

    # Opening
    if (dataset == 'traffic'):
        st_elem = se_open
    else:
        st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)

    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN, st_elem)

    # Tracking
    # Initialize tracker with first frame and bounding box
    morph = (morph * 255).astype('uint8')
    kalman = tracking.KalmanTracker(min_matches=5, stabilize_prediction=5)
    tracker_raw = []
    tracker_bin = []

    frame_no = 0
    for im_bin, im_raw in zip(morph, test):
        bboxes = tracking.find_bboxes(im_bin)
        kalman_pred = kalman.estimate(bboxes)
        out_raw = tracking.draw_tracking_prediction(im_raw, kalman_pred)
        out_bin = tracking.draw_tracking_prediction(im_bin[..., np.newaxis],
                                                    kalman_pred)
        # Append result
        tracker_raw.append(out_raw)
        tracker_bin.append(out_bin)

        # Some debug information
        frame_no += 1
        print('---------{:03d}---------'.format(frame_no))
        print(bboxes)
        for det in kalman_pred:
            print('>>', det)

    # Save individual gifs and an extra gif which compare them
    animations.video_recorder_v2(pred, f"{dataset}_bg.gif")
    animations.video_recorder_v2(morph, f"{dataset}_morph.gif")
    animations.video_recorder_v2(tracker_raw, f"{dataset}_track_raw.gif")
    animations.video_recorder_v2(tracker_bin, f"{dataset}_track_bin.gif")
