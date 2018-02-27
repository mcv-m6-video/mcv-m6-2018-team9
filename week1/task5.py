import os.path

import matplotlib.pyplot as plt
import cv2

from video import optical_flow


basedir = os.path.dirname(__file__)
TEST_IMAGE = os.path.join(basedir, 'resources', '000157_10.png')
OPT_FLOW_GT = os.path.join(basedir, 'resources', 'gt_000157_10.png')
OPT_FLOW_PRED = os.path.join(basedir, 'resources', 'LKflow_000157_10.png')


def run():
    """Week 1 - Task 5: Optical flow plotting

    - The original optical flow arrays have too many motion vectors to be
      represented in a plot.
    - Arrows might be confusing, since they are not related to pixels

    Propose a simplification method to better visualize optical flow results.

    """
    im = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
    gt = optical_flow.read_file(OPT_FLOW_GT)
    plt.figure()
    plt.title("Optical flow mask (Sequence 157 ground truth)")
    plt.imshow(gt[:,:,2])
    optical_flow.plot_map(im, gt, size=30,
                          title="Optical flow (Sequence 157 ground truth)")
    pred = optical_flow.read_file(OPT_FLOW_PRED)
    optical_flow.plot_map(im, pred, size=30,
                          title="Optical flow (Sequence 157 prediction)")
