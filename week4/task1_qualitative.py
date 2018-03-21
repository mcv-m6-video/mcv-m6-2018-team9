from data import kitti
from evaluation import metrics
from video import optical_flow


def run():
    """Plot qualitative results for optical flow prediction"""
    block_size = 40
    max_motion = 32

    # Kitti dataset, sequences 45 and 157
    seq45, gt45 = kitti.read_sequence(45, annotated=True)
    seq157, gt157 = kitti.read_sequence(157, annotated=True)

    # Sequence 45
    pred45 = optical_flow.block_matching(seq45[0], seq45[1],
                                         block_size=block_size,
                                         max_motion=max_motion)
    msen45, pepn45, err_im45, err_vec45 = metrics.optflow_metrics(pred45, gt45)

    # Sequence 157
    pred157 = optical_flow.block_matching(seq157[0], seq157[1],
                                          block_size=block_size,
                                          max_motion=max_motion)
    msen157, pepn157, err_im157, err_vec157 = metrics.optflow_metrics(pred157, gt157)

    # Plots for sequence 45
    metrics.plot_optflow_errors(err_vec45, err_im45,
                                f'sequence #45')
    optical_flow.plot_map(seq45[0], pred45, size=block_size,
                          title='Optical flow prediction (sequence #45)')
    print(f'MSEN for sequence 45: {msen45:0.02f}')
    print(f'PEPN for sequence 45: {pepn45:0.02f}')

    # Plots for sequence 157
    metrics.plot_optflow_errors(err_vec157, err_im157,
                                f'sequence #157')
    optical_flow.plot_map(seq157[0], pred157, size=block_size,
                          title='Optical flow prediction (sequence #157)')
    print(f'MSEN for sequence 157: {msen157:0.02f}')
    print(f'PEPN for sequence 157: {pepn157:0.02f}')
