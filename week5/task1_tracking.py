import numpy as np
import cv2
from data import cdnet
from evaluation import metrics, animations
from video import bg_subtraction, morphology, video_stabilization, tracking
import matplotlib.pyplot as plt
import time


compareOutput = 0
maxWidth = 320
out_path = '.'


def run(dataset):
    # Dataset-specific parameters
    if dataset == 'highway':
        rho = 0.20
        alpha = 2.8
        bsize = 50
        se_close = (15, 15)
        se_open = (4, 17)
        shadow_t1 = 0.082
        shadow_t2 = 0.017

    elif dataset == 'traffic':
        rho = 0.15
        alpha = 2.25
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
    train, gt_t = cdnet.read_sequence('week4', dataset, 'train',
                                      colorspace='gray',annotated=True)
    test, gt = cdnet.read_sequence('week4', dataset, 'test', colorspace='gray',
                                   annotated=True)

    # Stabilize sequences
    # TODO: check from previous weeks if video stabilization needs color images
    train_stab, train_mask = video_stabilization.ngiaho_stabilization(train,
                                    gt_t, out_path, compareOutput, maxWidth)
    test_stab, test_mask = video_stabilization.ngiaho_stabilization(test,gt,
                                    out_path, compareOutput, maxWidth)

    # Add axis
    test_stab = test_stab[...,np.newaxis]
    train_stab = train_stab[...,np.newaxis]

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    model_stab = bg_subtraction.create_model_mask(train_stab, train_mask[1,:])

    pred = bg_subtraction.predict(test, model, alpha, rho=rho)
    pred_stab = bg_subtraction.predict_masked(test_stab, test_mask[1],
                                              model_stab, alpha, rho=rho)

    # Imfill + filter small blobs
    filled8 = morphology.imfill(pred, neighb=8)
    filled8_stab = morphology.imfill(pred_stab, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)
    clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

    # Closing
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_close)
    clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE, st_elem)
    clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE, st_elem)

    # Opening
    if (dataset == 'traffic'):
        st_elem = se_open
    else:
        st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)

    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN, st_elem)
    morph_stab = morphology.filter_morph(clean_stab, cv2.MORPH_OPEN, st_elem)

    # Tracking
    # Initialize tracker with first frame and bounding box
    morph = (morph * 255).astype('uint8')
    kalman = tracking.Tracker()
    colors = [(255, 0, 0), (255, 255, 0), (255, 255, 255), (255, 0, 255),
              (0, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
    tracker_out = []

    for im_bin, im_raw in zip(morph, test):
        bboxes = tracking.find_bboxes(im_bin)
        kalman_pred = kalman.estimate(bboxes)
        out_im = tracking.draw_tracking_prediction(im_raw, kalman_pred)

        print('-----------------')
        print(bboxes)

        # Append result
        tracker_out.append(out_im)

    tracker_out = np.array(tracker_out)

    # Save individual gifs and an extra gif which compare them
    animations.video_recorder_v2(pred, f"{dataset}_orig.gif")
    animations.video_recorder_v2(morph, f"{dataset}_morph.gif")
    animations.video_recorder_v2(test_mask[1,:], f"{dataset}_valid.gif")
    animations.video_recorder_v2(tracker_out, f"{dataset}_track.gif")
