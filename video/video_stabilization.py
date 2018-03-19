import cv2
import imutils
import pandas as pd
import numpy as np


def ngiaho_stabilization(test, gt, out_path='.', compare_output=1,
                         max_width=640):
    """

    :param test:
    :param out_path:
    :param compare_output:
    :param max_width:
    :return:
    """
    n, h, w, _ = test.shape


    # initialize storage
    prev_to_cur_transform = []

    prev_im = test[0]
    #prev_gray = cv2.cvtColor(prev_im, cv2.COLOR_RGB2GRAY)
    prev_gray = test[0]
    # iterate through frame count
    for k in range(1, n):
        # read current frame
        curr_im = test[k]
        # convert to gray
        #curr_gray = cv2.cvtColor(curr_im, cv2.COLOR_RGB2GRAY)
        curr_gray = curr_im
        # use GFTT for keypoint detection
        prev_corner = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                              qualityLevel=0.01,
                                              minDistance=30.0, blockSize=3)
        # calc flow of movement (resource: http://opencv-python-tutroals.
        # readthedocs.io/en/latest/py_tutorials/py_video/
        # py_lucas_kanade/py_lucas_kanade.html)
        cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                           prev_corner, None)
        # storage for keypoints with status 1
        prev_corner2 = []
        cur_corner2 = []
        for i, st in enumerate(status):
            # if keypoint found in frame i & i-1
            if st == 1:
                # store coords of keypoints that appear in both
                prev_corner2.append(prev_corner[i])
                cur_corner2.append(cur_corner[i])
        prev_corner2 = np.array(prev_corner2)
        cur_corner2 = np.array(cur_corner2)
        # estimate partial transform (resource: http://nghiaho.com/?p=2208)
        T_new = cv2.estimateRigidTransform(prev_corner2, cur_corner2, False)
        if T_new is not None:
            T = T_new
        # translation x
        dx = T[0, 2]
        # translation y
        dy = T[1, 2]
        # rotation
        da = np.arctan2(T[1, 0], T[0, 0])
        # store for saving to disk as table
        prev_to_cur_transform.append([dx, dy, da])
        # set current frame to prev frame for use in next iteration
        prev_im = curr_im[:]
        prev_gray = curr_gray[:]

    # convert list of transforms to array
    prev_to_cur_transform = np.array(prev_to_cur_transform)
    # cumsum of all transforms for trajectory
    trajectory = np.cumsum(prev_to_cur_transform, axis=0)

    # convert trajectory array to df
    trajectory = pd.DataFrame(trajectory)
    # rolling mean to smooth
    smoothed_trajectory = trajectory.rolling(window=30, center=False).mean()
    # back fill nas caused by smoothing
    smoothed_trajectory = smoothed_trajectory.fillna(method='bfill')
    # save smoothed trajectory
    smoothed_trajectory.to_csv('{}/smoothed.csv'.format(out_path))

    # new set of prev to cur transform, removing trajectory and replacing w/smoothed
    new_prev_to_cur_transform = prev_to_cur_transform + (
            smoothed_trajectory - trajectory)
    # write transforms to disk
    new_prev_to_cur_transform.to_csv(
        '{}/new_prev_to_cur_transformation.csv'.format(out_path))

    #####
    # APPLY VIDEO STAB
    #####
    # initialize transformation matrix
    T = np.zeros((2, 3))
    # convert transform df to array
    new_prev_to_cur_transform = np.array(new_prev_to_cur_transform)
    # setup video cap
    # cap = cv2.VideoCapture(args['video'])
    # set output width based on option for saving old & stabilized video side by side
    w_write = min(w, max_width)
    # correct height change caused by width change
    h_write = imutils.resize(prev_gray, width=w_write).shape[0]
    # double output width if option chosen for side by side comparison
    if compare_output > 0:
        w_write = w_write * 2
    # setup video writer
    # out = cv2.VideoWriter('{}/stabilized_output.avi'.format(out_path),
    #                      cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), fps,
    #                      (w_write, h_write), True)

    out = []
    out_mask_gt = []
    out_mask_valid = []
    # loop through frame count
    for k in range(0, n-1):
        # read current frame
        cur = test[k]
        # read/build transformation matrix
        T[0, 0] = np.cos(new_prev_to_cur_transform[k][2])
        T[0, 1] = -np.sin(new_prev_to_cur_transform[k][2])
        T[1, 0] = np.sin(new_prev_to_cur_transform[k][2])
        T[1, 1] = np.cos(new_prev_to_cur_transform[k][2])
        T[0, 2] = new_prev_to_cur_transform[k][0]
        T[1, 2] = new_prev_to_cur_transform[k][1]
        # apply saved transform (resource: http://nghiaho.com/?p=2208)
        cur2 = cv2.warpAffine(cur, T, (w, h))
        mask_gt = cv2.warpAffine(np.array(gt[k,0],dtype=np.uint8), T, (w, h))
        mask_valid = cv2.warpAffine(np.array(gt[k,1],dtype=np.uint8), T, (w, h))
        # build side by side comparison if option chosen
        if compare_output > 0:
            # resize to maxwidth (if current width larger than max_width)
            cur_resize = imutils.resize(cur, width=min(w, max_width))
            # resize to maxwidth (if current width larger than maxwidth)
            cur2_resize = imutils.resize(cur2, width=min(w, max_width))
            # combine arrays for side by side
            cur2 = np.hstack((cur_resize, cur2_resize))
        else:
            # resize to maxwidth (if current width larger than maxwidth)
            cur2 = imutils.resize(cur2, width=min(w, max_width))
            mask_gt = imutils.resize(mask_gt, width=min(w, max_width))
            mask_valid = imutils.resize(mask_valid, width=min(w, max_width))
        # show frame to screen
        # cv2.imshow('stable', cur2)
        # cv2.waitKey(20)
        # write frame to output video
        # out.write(cur2)
        out.append(cur2)
        out_mask_gt.append(mask_gt)
        out_mask_valid.append(mask_valid)

    print(test[n-1,:,:,0].shape)
    out.append(test[n-1,:,:,0])
    out_mask_gt.append(gt[n-1,0])
    out_mask_valid.append(gt[n-1,1])

    out = np.array(out)
    out_mask_gt = np.array(out_mask_gt)
    out_mask_valid = np.array(out_mask_valid)

    print(out_mask_valid.shape)
    print(out_mask_gt.shape)
    out_gt = np.array([out_mask_gt,out_mask_valid])
    return out, out_gt