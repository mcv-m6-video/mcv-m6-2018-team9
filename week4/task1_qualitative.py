from data import kitti
from evaluation import metrics
from video import optical_flow


def run():
    """Plot qualitative results for optical flow prediction"""
    block_size = 32
    max_motion = 48
    compensation = 'backward'

    # Set indices depending on backward or forward compensation
    if compensation == 'forward':
        a, b = (0, 1)
    else:
        a, b = (1, 0)

    # Kitti dataset, sequences 45 and 157
    seq45, gt45 = kitti.read_sequence(45, annotated=True)
    seq157, gt157 = kitti.read_sequence(157, annotated=True)

    # Sequence 45
    pred45 = optical_flow.block_matching(seq45[a], seq45[b],
                                         block_size=block_size,
                                         max_motion=max_motion)
    __, pepn45, err_im45, err_vec45 = metrics.optflow_metrics(pred45, gt45)

    # Sequence 157
    pred157 = optical_flow.block_matching(seq157[a], seq157[b],
                                          block_size=block_size,
                                          max_motion=max_motion)
    __, pepn157, err_im157, err_vec157 = metrics.optflow_metrics(pred157, gt157)

    if compensation == 'backward':
        # Change to opposite direction (we plot image 0)
        pred45 = -pred45
        pred157 = -pred157

    # Plots for sequence 45
    metrics.plot_optflow_errors(err_vec45, err_im45,
                                f'sequence #45, PEPN={pepn45:0.02f}')
    optical_flow.plot_map(seq45[0], pred45, size=block_size,
                          title='Optical flow prediction (sequence #45)')

    # Plots for sequence 157
    metrics.plot_optflow_errors(err_vec157, err_im157,
                                f'sequence #157, PEPN={pepn157:0.02f}')
    optical_flow.plot_map(seq157[0], pred157, size=block_size,
                          title='Optical flow prediction (sequence #157)')
