import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import kitti
from evaluation import metrics
from video import optical_flow


def run():
    """Week 4, Task 1 - Choose the best block size and area size

    Performs a grid search for hyperparameter selection in the block matching
    algorithm for optical flow estimation.

    PCPN (percentage of correct pixels in non-occluded areas, PEPN - 1) is
    chosen as the criterion to determine the best model parameters.

    """
    # Hyperparameters
    block_size = [4, 8, 16, 32, 48, 64]
    max_motion = [4, 8, 16, 32, 48, 64]
    compensation = 'forward'

    # Set indices depending on backward or forward compensation
    if compensation == 'forward':
        a, b = (0, 1)
    else:
        a, b = (1, 0)

    # Kitti dataset, sequences 45 and 157
    seq45, gt45 = kitti.read_sequence(45, annotated=True)
    seq157, gt157 = kitti.read_sequence(157, annotated=True)

    # Grid search
    pepn = np.zeros([len(block_size), len(max_motion)], dtype='float32')
    for i, bsize in enumerate(block_size):
        for j, maxm in enumerate(max_motion):
            # Sequence 45
            pred45 = optical_flow.block_matching(seq45[a], seq45[b],
                                                 block_size=bsize,
                                                 max_motion=maxm)
            __, pepn45, __, __ = metrics.optflow_metrics(pred45, gt45)

            # Sequence 157
            pred157 = optical_flow.block_matching(seq157[a], seq157[b],
                                                 block_size=bsize,
                                                 max_motion=maxm)
            __, pepn157, __, __ = metrics.optflow_metrics(pred157, gt157)

            # Store mean value
            pepn[i, j] = (pepn45 + pepn157) / 2
            print(f'bsize {block_size[i]}, maxm {max_motion[j]}: '
                  f'{pepn[i,j]:0.02f}')


    # Find best score
    pcpn = 100 - pepn
    i, j = np.unravel_index(pcpn.argmax(), pcpn.shape)
    print(f'Best parameters for {compensation} compensation:')
    print(f'- Block size: {block_size[i]}')
    print(f'- Max motion: {max_motion[j]}')
    print(f'With a PEPN = {pepn[i,j]:0.02f} (PCPN = {pcpn[i,j]:0.02f})')

    # Plot surface of results
    metrics.plot_gridsearch_3d(block_size, max_motion, pcpn,
                               xlabel='Block size', ylabel='Max motion',
                               zlabel='PCPN', title='Grid search')
