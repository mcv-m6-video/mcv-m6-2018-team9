import os.path
import matplotlib as mpl
mpl.use('TkAgg')


from video import optical_flow
from evaluation import metrics


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
    VecErr1, MSEN1, PEPN1 = metrics.msen(test1,gt1)

    print("MSEN & PEPN for sequence 157")
    test2 = optical_flow.read_file(TEST_FLOW_2)
    gt2 = optical_flow.read_file(GT_FLOW_2)
    VecErr2, MSEN2, PEPN2 = metrics.msen(test2,gt2)
