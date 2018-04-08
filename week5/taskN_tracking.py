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
    H = Homography.DLT(coords)

    se_open = np.eye(l, dtype=np.uint8)
    for r in range(0, k):
        se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r + 1))
        se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r - 1))
    se_open = np.transpose(se_open.astype(np.uint8))

    se_open = (5,5)

    rho = 0.15

    #TRACKING TYPE: 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN'
    # Read dataset
    # train, gt_t = cdnet.read_sequence('week4', dataset, 'train',
    #                             colorspace='gray',annotated=True)
    # test, gt = cdnet.read_sequence('week4', dataset, 'test',
    #                                colorspace='gray', annotated=True)

    seq = workshop.read_sequence(dataset, colorspace='rgb', homography=H)

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
    animations.video_recorder(train, '', f"{dataset}_train_stab")
    animations.video_recorder(test, '', f"{dataset}_test_stab")
    model = bg_subtraction.create_model(train)
    # model_stab = bg_subtraction.create_model_mask(train_stab,
    #                                               train_mask[1,:])


    pred = bg_subtraction.predict(test, model, alpha2, rho=rho)
    # pred_stab = bg_subtraction.predict_masked(test_stab, test_mask[1],
    #                                     model_stab, alpha2, rho=rho[dataset])


    filled8 = morphology.imfill(pred, neighb=8)
    # filled8_stab = morphology.imfill(pred_stab, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)
    # clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

    # CLOSING
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_close)
    #clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
    #                                st_elem)
    # clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE,
    #                                      st_elem)

    # OPENING

    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_open)

    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN,
                                    st_elem)
    # morph_stab = morphology.filter_morph(clean_stab, cv2.MORPH_OPEN,
    #                                      st_elem)

    # TRACKING
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    #params.minArea = 10000

    detector = cv2.SimpleBlobDetector_create(params)
    morph = (morph*255).astype('uint8')

    # Initialize tracker with first frame and bounding box
    trk = tracking.Tracker(3, max_distance=200)
    colors = [(255, 0, 0), (255, 255, 0), (255, 255, 255), (255, 0, 255),
              (0, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
    tracker_out = []

    animations.video_recorder(pred, '', f"{dataset}_orig")

    for idx in range(1, morph.shape[0]):
        if True:
            # Read a new frame
            frame = morph[idx]
            out_im = test[idx]

            # Start timer
            timer = cv2.getTickCount()
            blob = detector.detect(frame)
            bboxes = np.array([(bb.pt[0], bb.pt[1], bb.pt[0] + bb.size / 2,
                                bb.pt[1] + bb.size / 2) for bb in blob])

            res_bboxes = trk.estimate(bboxes)

            # Draw bounding box
            for i, cbb in enumerate(res_bboxes):
                # Tracking success
                p1 = (int(cbb['location'][0] - cbb['height']),
                      int(cbb['location'][1] - cbb['width']))
                p2 = (int(cbb['location'][0] + cbb['height']),
                      int(cbb['location'][1] + cbb['width']))
                cv2.rectangle(out_im, p1, p2, colors[cbb['id'] % 8], 2, 1)

            # Append result
            tracker_out.append(out_im)

            # Exit if ESC pressed
            k = cv2.waitKey(50) & 0xff
            if k == 27: break

        #except Exception as e:
        else:
            pass
            # print(e)

    tracker_out = np.array(tracker_out)

    # Save individual gifs and an extra gif which compare them
    animations.video_recorder(pred, '', f"{dataset}_orig")
    animations.video_recorder(morph, '', f"{dataset}_morph")
    animations.video_recorder(tracker_out, '', f"{dataset}_track")