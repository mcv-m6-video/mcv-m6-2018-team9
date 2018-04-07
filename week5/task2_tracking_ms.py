"""Mean-Shift tracking based on OpenCV with Python Blueprints
     Implementation based on Visually Salient Objects in a video sequence.
"""

import cv2
import numpy as np
from os import path
from data import cdnet
from evaluation import metrics, animations
from video import bg_subtraction, morphology, optical_flow, tracking
from week5 import saliency


compareOutput = 0
maxWidth = 320
out_path = '.'

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def run(dataset):

    # Model & Morphologyparameters
    if dataset == 'traffic':
        bsize = 400
        alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                       np.linspace(11, 40, 10)])
        se_close = (15, 15)
        k = 9
        l = 30
        #alpha2 = 1.38
        alpha2=2.24
        rho = 0.15
        #TRACKING PARAMS
        #join th threshold to group rectangles. 0 -> keep all / inf -> keep 1
        #in traffic we are only ev
        join_th = 10000
        gk=(5,5)
        min_area = 1500
        min_shift2 = 5

        se_open = np.eye(l, dtype=np.uint8)
        for r in range(0, k):
            se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r + 1))
            se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r - 1))
        se_open = np.transpose(se_open.astype(np.uint8))

    if dataset == 'highway':
        rho = 0.20
        alpha2 = 2.8
        bsize = 50
        se_close = (15, 15)
        se_open = (4, 17)
        shadow_t1 = 0.082
        shadow_t2 = 0.017
        #TRACKING PARAMS
        join_th = 0
        gk = (3, 3)
        min_area = 400
        min_shift2 = 3

    # Read dataset
    train, gt_t = cdnet.read_sequence('week4', dataset, 'train',
                                colorspace='rgb',annotated=True)
    test, gt = cdnet.read_sequence('week4', dataset, 'test',
                                   colorspace='rgb', annotated=True)

    # Stabilize sequences
    train_stab, train_mask = optical_flow.stabilize(train, mode='f')
    test_stab, test_mask,new_gt = optical_flow.stabilize(test, gt, mode='f')

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    model_stab = bg_subtraction.create_model_mask(train_stab, train_mask)


    pred = bg_subtraction.predict(test, model, alpha2,rho=rho)
    pred_stab = bg_subtraction.predict_masked(test_stab, test_mask,
                                        model_stab, alpha2, rho=rho)

    filled8 = morphology.imfill(pred, neighb=8)
    filled8_stab = morphology.imfill(pred_stab, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)
    clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

    # CLOSING
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_close)
    clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE, st_elem)
    clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE, st_elem)

    # OPENING
    if (dataset == 'traffic'):
        st_elem = se_open
    else:
        st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)

    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN,st_elem)
    morph_stab = morphology.filter_morph(clean_stab, cv2.MORPH_OPEN,st_elem)

    if(dataset == 'traffic'):
        seq_track = (morph_stab * 255).astype('uint8')
    else:
        seq_track = (morph*255).astype('uint8')

    mot = tracking.MultipleObjectsTracker(min_area,min_shift2)

    tracker_out = []

    # Initialize tracker with first frame and bounding box

    for idx in range(1, morph_stab.shape[0]):
        # Read a new frame
        #frame = morph[idx]
        img = seq_track[idx]
        out_im = test_stab[idx]

        # generate saliency map
        sal = saliency.Saliency(img, use_numpy_fft=False, gauss_kernel=gk)

        cv2.imshow('original', img)
        cv2.imshow('saliency', sal.get_saliency_map())
        cv2.imshow('objects', sal.get_proto_objects_map(use_otsu=False))
        cv2.imshow('tracker', mot.advance_frame(out_im,
                   sal.get_proto_objects_map(use_otsu=False),join_th))
        im_out_tr = mot.advance_frame(out_im,
                                    sal.get_proto_objects_map(use_otsu=False))
        tracker_out.append(im_out_tr)

    tracker_out = np.array(tracker_out)

    # Save individual gifs and an extra gif which compare them
    animations.video_recorder(pred, '', f"{dataset}_orig")
    animations.video_recorder(morph, '', f"{dataset}_morph")
    animations.video_recorder(morph_stab, '', f"{dataset}_morph_stab")
    animations.video_recorder(gt[1,:], '', f"{dataset}_valid")
    animations.video_recorder(tracker_out, '', f"{dataset}_track")
