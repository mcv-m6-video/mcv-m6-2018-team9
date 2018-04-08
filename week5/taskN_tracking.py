#script to perform video stabilization based on AdamSpannbauer implementation.
#	INPUTS: IMAGE SEQUENCE
#	OUTPUTS: STABELIZED VIDEO AVI, TRANSFORMATION DF CSV, SMOOTHED TRAJECTORY CSV

#RESOURCE & ORIGINAL CPP AUTHOR OF THIS VIDEO STAB LOGIC: http://nghiaho.com/?p=2093
#ORIGINAL CPP: http://nghiaho.com/uploads/code/videostab.cpp

import numpy as np
import cv2
from data import workshop
from evaluation import metrics, animations
from video import bg_subtraction, morphology, video_stabilization, tracking, \
    optical_flow, Homography
import matplotlib.pyplot as plt
import time


compareOutput = 0
maxWidth = 320
out_path = '.'

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def run(dataset):

    if dataset == 'sequence1':
        marker = 150
    elif dataset == 'sequence2':
        marker = 80
    elif dataset == 'sequence3':
        marker = 100
    else:
        raise Exception('Wrong dataset')

    # Model & Morphologyparameters
    bsize = 100
    alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                   np.linspace(11, 40, 10)])
    se_close = (3, 3)
    k = 9
    l = 30
    alpha2 = 1.38

    coords = [(130, 23), (160, 23), (95, 138),
              (225, 160)]

    se_open = np.eye(l, dtype=np.uint8)
    for r in range(0, k):
        se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r + 1))
        se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r - 1))
    se_open = np.transpose(se_open.astype(np.uint8))

    se_open = (20, 3)

    rho = 0.15

    #TRACKING TYPE: 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN'
    # Read dataset
    # train, gt_t = cdnet.read_sequence('week4', dataset, 'train',
    #                             colorspace='gray',annotated=True)
    # test, gt = cdnet.read_sequence('week4', dataset, 'test',
    #                                colorspace='gray', annotated=True)

    seq = workshop.read_sequence(dataset, colorspace='rgb')

    train = seq[:marker]
    test = seq[marker:]

    # Stabilize sequences
    # train_stab, train_mask = video_stabilization.ngiaho_stabilization(train,
    #                                 gt_t, out_path, compareOutput, maxWidth)
    # test_stab, test_mask = video_stabilization.ngiaho_stabilization(test,gt,
    #                                 out_path, compareOutput, maxWidth)

    # Add axis
    # test_stab = test_stab[...,np.newaxis]
    # train_stab = train_stab[...,np.newaxis]

    # Adaptive model prediction
    train = train.astype(np.uint8)
    test = test.astype(np.uint8)
    animations.video_recorder(train, '', f"{dataset}_train_unstab")
    animations.video_recorder(test, '', f"{dataset}_test_unstab")
    train, __ = optical_flow.stabilize(train, mode='f')
    test, __ = optical_flow.stabilize(test, mode='f')
    train = Homography.DLT(train, coords)
    test = Homography.DLT(test, coords)
    train = train.astype(np.uint8)
    test = test.astype(np.uint8)
    train, mask_train = optical_flow.stabilize(train, mode='f')
    test, mask_test = optical_flow.stabilize(test, mode='f')
    animations.video_recorder(train, '', f"{dataset}_train_stab")
    animations.video_recorder(test, '', f"{dataset}_test_stab")
    model = bg_subtraction.create_model_mask(train, mask_train[1, :])
    # model_stab = bg_subtraction.create_model_mask(train_stab,
    #                                               train_mask[1,:])


    pred = bg_subtraction.predict_masked(test, mask_test[1, :], model, alpha2, rho=rho)
    # pred_stab = bg_subtraction.predict_masked(test_stab, test_mask[1],
    #                                     model_stab, alpha2, rho=rho[dataset])


    filled8 = morphology.imfill(pred, neighb=8)
    # filled8_stab = morphology.imfill(pred_stab, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)
    # clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

    # CLOSING
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_close)
    clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
                                   st_elem)
    # clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE,
    #                                      st_elem)

    # OPENING

    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_open)

    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN,
                                    st_elem)
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
