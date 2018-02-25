import os.path

import matplotlib.pyplot as plt
import cv2

from video import optical_flow


basedir = os.path.dirname(__file__)
TEST_IMAGE_1 = os.path.join(basedir, 'resources', '000045_10.png')
TEST_IMAGE_2 = os.path.join(basedir, 'resources', '000157_10.png')
OPT_FLOW_1 = os.path.join(basedir, 'resources', 'gt_000045_10.png')
OPT_FLOW_2 = os.path.join(basedir, 'resources', 'gt_000157_10.png')


def run():
    """Week 1 - Task 5: Optical flow plotting

    - The original optical flow arrays have too many motion vectors to be
      represented in a plot.
    - Arrows might be confusing, since they are not related to pixels

    Propose a simplification method to better visualize optical flow results.

    """
    im1 = cv2.imread(TEST_IMAGE_1, cv2.IMREAD_GRAYSCALE)
    flow1 = optical_flow.read_file(OPT_FLOW_1)
    plt.figure()
    plt.title("Optical flow mask (Image 45)")
    plt.imshow(flow1[:,:,2])
    optical_flow.plot_map(im1, flow1, size=30,
                          title="Optical flow map (Image 45)")

    im2 = cv2.imread(TEST_IMAGE_2, cv2.IMREAD_GRAYSCALE)
    flow2 = optical_flow.read_file(OPT_FLOW_2)
    plt.figure()
    plt.title("Optical flow mask (Image 157)")
    plt.imshow(flow2[:,:,2])
    optical_flow.plot_map(im2, flow2, size=30,
                          title="Optical flow map (Image 157)")
