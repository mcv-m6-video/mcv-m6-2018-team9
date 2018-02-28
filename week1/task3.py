import os.path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from video import optical_flow
from evaluation import metrics
from matplotlib.ticker import FuncFormatter


basedir = os.path.dirname(__file__)
TEST_FLOW_1 = os.path.join(basedir, 'resources', 'LKflow_000045_10.png')
TEST_FLOW_2 = os.path.join(basedir, 'resources', 'LKflow_000157_10.png')
GT_FLOW_1 = os.path.join(basedir, 'resources', 'gt_000045_10.png')
GT_FLOW_2 = os.path.join(basedir, 'resources', 'gt_000157_10.png')

def run():
    """Week 1 - Task 3: MSEN & PEPN


    """
    print("MSEN & PEPN for sequence 45")
    test1 = optical_flow.read_file(TEST_FLOW_1)
    gt1 = optical_flow.read_file(GT_FLOW_1)
    msen_45, pepn_45, err_img_45, err_vect_45 = metrics.msen(test1,gt1)

    print('PEPN: ', pepn_45)
    print('MSEN: ', msen_45)

    plt.figure(1)
    cm = plt.cm.get_cmap('viridis')
    n, bins, patches = plt.hist(err_vect_45, bins=25, normed=1,stacked=False)
    bins = 100*bins/np.sum(bins)


    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    plt.xlabel('Motion Error Magintude')
    plt.ylabel('Nº of Pixels')
    plt.title(" MSEN  00045_10.png")
    plt.show()

    plt.figure(2)
    plt.title('Motion Error Image 000157_10.png reconstruction')
    plt.imshow(err_img_45)
    plt.colorbar()
    plt.show()

    #IMAGE 157

    print("MSEN & PEPN for sequence 157")
    test2 = optical_flow.read_file(TEST_FLOW_2)
    gt2 = optical_flow.read_file(GT_FLOW_2)
    msen_157, pepn_157, err_img_157, err_vect_157 = metrics.msen(test2,gt2)

    print('PEPN: ', pepn_157)
    print('MSEN: ', msen_157)

    plt.figure(3)
    cm = plt.cm.get_cmap('viridis')

    # Plot histogram.
    n, bins, patches = plt.hist(err_vect_157, bins=25, stacked=False)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))


    plt.xlabel('Motion Error Magintude')
    plt.ylabel('Nº of Pixels')
    plt.title(" MSEN 00157_10.png")
    plt.show()

    plt.figure(4)
    plt.title('Motion Error Image 000157_10.png reconstruction')
    plt.imshow(err_img_157)
    plt.colorbar()
    plt.show()
