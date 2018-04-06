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
        alpha2 = 1.38

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


    #TRACKING TYPE: 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN'
    tracker_type = 'KCF'
    print('tracker type: ', str(tracker_type))

    # Read dataset
    train, gt_t = cdnet.read_sequence('week4', dataset, 'train',
                                colorspace='gray',annotated=True)
    test, gt = cdnet.read_sequence('week4', dataset, 'test',
                                   colorspace='gray', annotated=True)

    # Stabilize sequences
    train_stab, train_mask = video_stabilization.ngiaho_stabilization(train,
                                    gt_t, out_path, compareOutput, maxWidth)
    test_stab, test_mask = video_stabilization.ngiaho_stabilization(test,gt,
                                    out_path, compareOutput, maxWidth)

    # Add axis
    test_stab = test_stab[...,np.newaxis]
    train_stab = train_stab[...,np.newaxis]

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    model_stab = bg_subtraction.create_model_mask(train_stab,
                                                  train_mask[1,:])


    pred = bg_subtraction.predict(test, model, alpha2,
                                  rho=rho)
    pred_stab = bg_subtraction.predict_masked(test_stab, test_mask[1],
                                        model_stab, alpha2, rho=rho)


    filled8 = morphology.imfill(pred, neighb=8)
    filled8_stab = morphology.imfill(pred_stab, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)
    clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

    # CLOSING
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        se_close)
    clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
                                    st_elem)
    clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE,
                                         st_elem)

    # OPENING
    if (dataset == 'traffic'):
        st_elem = se_open
    else:
        st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)

    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN,
                                    st_elem)
    morph_stab = morphology.filter_morph(clean_stab, cv2.MORPH_OPEN,
                                         st_elem)

    # TRACKING

    if int(minor_ver) < 3:
        print('yuhu')
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    morph = (morph*255).astype('uint8')

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    # params.minArea = 10000

    im_disp, bbox = tracking.detect_blobs(morph[0,:,:],test_stab[0],params)

    tracker_out = [im_disp]

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(test[0], bbox)

    for idx in range(1, morph.shape[0]):
        # Read a new frame
        #frame = morph[idx]
        frame = test[idx]
        out_im = test[idx]

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)
        #####dcsevrrzdgxgfnhmj,kl.kjhmgnfdgefsadws

        # Calculate Frames per second (FPS)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        fps = 12

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(out_im, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(out_im, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(out_im, tracker_type + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(out_im, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Append result
        tracker_out.append(out_im)

        # Exit if ESC pressed
        k = cv2.waitKey(50) & 0xff
        if k == 27: break

    tracker_out = np.array(tracker_out)

    # Save individual gifs and an extra gif which compare them
    animations.video_recorder(pred, '', f"{dataset}_orig")
    animations.video_recorder(test_mask[1,:], '', f"{dataset}_valid")
    animations.video_recorder(tracker_out, '', f"{dataset}_track")