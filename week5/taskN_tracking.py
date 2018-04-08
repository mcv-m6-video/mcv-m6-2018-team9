import numpy as np
import cv2

from data import workshop
from evaluation import animations
from video import (bg_subtraction, morphology, video_stabilization, tracking,
                   optical_flow, Homography)


def run(dataset):

    if dataset == 'sequence1':
        marker = 150
    elif dataset == 'sequence2':
        marker = 80
    elif dataset == 'sequence3':
        marker = 100
    else:
        raise Exception('Wrong dataset')

    # Background subtraction model
    rho = 0.15
    alpha = 4

    # Morphology parameters
    bsize = 100
    se_close = (3, 3)
    k = 9
    l = 30

    se_open = np.eye(l, dtype=np.uint8)
    for r in range(0, k):
        se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r + 1))
        se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r - 1))
    se_open = np.transpose(se_open.astype(np.uint8))

    se_open = (5,5)

    # Area for perspective correction and speed estimation
    detection_area = [(130, 23), (160, 23), (85, 160), (225, 160)]

    # Read dataset
    seq = workshop.read_sequence(dataset, colorspace='gray')
    train = seq[:marker]
    test = seq[marker:]

    # Video stabilization and adaptive model prediction
    train, train_mask = optical_flow.stabilize(train, mode='forward')
    test, test_mask = optical_flow.stabilize(test, mode='forward')

    # train = Homography.DLT(train, coords)
    # test = Homography.DLT(test, coords)

    animations.video_recorder(train, '', f"{dataset}_train_stab")
    animations.video_recorder(test, '', f"{dataset}_test_stab")

    model = bg_subtraction.create_model_mask(train, train_mask)
    pred = bg_subtraction.predict_masked(test, test_mask, model, alpha,
                                         rho=rho)

    # Morphology and filtering
    filled8 = morphology.imfill(pred, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)

    # CLOSING
    # st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_close)
    # clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
    #                                st_elem)
    # clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE,
    #                                      st_elem)

    # OPENING
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_open)
    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN, st_elem)
    # morph_stab = morphology.filter_morph(clean_stab, cv2.MORPH_OPEN,
    #                                      st_elem)

    # Kalman tracker
    disappear_thr = 3
    min_matches = 3
    stabilize_prediction = 5
    morph = (morph*255).astype('uint8')
    kalman = tracking.KalmanTracker(disappear_thr=disappear_thr,
                                    min_matches=min_matches,
                                    stabilize_prediction=stabilize_prediction)
    tracker_raw = []
    tracker_bin = []
    frame_no = 0

    for im_bin, im_raw in zip(morph, test):
        bboxes = tracking.find_bboxes(im_bin)
        kalman_pred = kalman.estimate(bboxes)
        out_raw = tracking.draw_tracking_prediction(im_raw, kalman_pred)
        out_bin = tracking.draw_tracking_prediction(im_bin[..., np.newaxis],
                                                    kalman_pred)

        # roi = [detection_area[0], detection_area[1], detection_area[3],
        #        detection_area[2]]
        # roi = np.reshape(roi, (1, -1, 1, 2))
        # out_raw = cv2.polylines(out_raw, roi, True, (240, 30, 0), 1,
        #                         cv2.LINE_AA)

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
    animations.video_recorder(pred, '', f"{dataset}_orig")
    animations.video_recorder(morph, '', f"{dataset}_morph")
    animations.video_recorder_v2(tracker_raw, f"{dataset}_track_raw.gif")
    animations.video_recorder_v2(tracker_bin, f"{dataset}_track_bin.gif")
