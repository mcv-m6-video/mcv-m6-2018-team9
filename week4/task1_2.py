import numpy as np

from data import kitti
from video import optical_flow
from evaluation import metrics

SEQ_ID_1 = 45
SEQ_NAME_1 = 'Sequence 45'

SEQ_ID_2 = 157
SEQ_NAME_2 = 'Sequence 157'


def run():

    sequences_id = [SEQ_ID_1, SEQ_ID_2]
    sequences_names = [SEQ_NAME_1, SEQ_NAME_2]

    results = []

    for seq_id, seq_name in zip(sequences_id, sequences_names):

        seq, gt = kitti.read_sequence(seq_id, annotated=True)

        flow_lk = optical_flow.lk_sequence(seq, valid_mask=gt[:, :, 2],
                                           wsize=(15, 15))

        flow_lk_pyr = optical_flow.lk_pyr_sequence(seq, valid_mask=gt[:, :, 2],
                                                   wsize=(15, 15), pyr_levels=3)

        flow_fnb = optical_flow.farneback_sequence(seq, levels=3, pyr_sc=0.5,
                                                   wsize=15, n_iter=10,
                                                   poly_n=7, p_sigma=1.5,
                                                   gauss=True)

        flow_lk = flow_lk[0]
        flow_lk_pyr = flow_lk_pyr[0]
        flow_fnb = np.concatenate((flow_fnb[0], np.ones((flow_fnb.shape[1],
                                                         flow_fnb.shape[2],
                                                         1))),
                                  axis=2)

        results.append(dict(title=seq_name,
                            lk=dict(flow=flow_lk,
                                    metrics=metrics.optflow_metrics(flow_lk,
                                                                    gt)),
                            lkpyr=dict(flow=flow_lk_pyr,
                                       metrics=metrics.optflow_metrics(
                                           flow_lk_pyr, gt)),
                            fnb=dict(flow=flow_fnb,
                                     metrics=metrics.optflow_metrics(flow_fnb,
                                                                     gt))))

    for res in results:
        print('Sequence ' + res['title'] + ':')
        print('=================')
        print(' Lucas Kanade: MMEN = ' + str(res['lk']['metrics'][0])
              + ', PEPN = ' + str(res['lk']['metrics'][1]))
        print(' Multiresolution Lucas Kanade: MMEN = '
              + str(res['lkpyr']['metrics'][0]) + ', PEPN = '
              + str(res['lkpyr']['metrics'][1]))
        print(' Farneback: MMEN = ' + str(res['fnb']['metrics'][0])
              + ', PEPN = ' + str(res['fnb']['metrics'][1]))

        metrics.plot_optflow_errors(res['lk']['metrics'][3],
                                    res['lk']['metrics'][2],
                                    res['title'] + ': Lucas-Kanade')

        metrics.plot_optflow_errors(res['lkpyr']['metrics'][3],
                                    res['lkpyr']['metrics'][2],
                                    res['title']
                                    + ': Multiresolution Lucas-Kanade')

        metrics.plot_optflow_errors(res['fnb']['metrics'][3],
                                    res['fnb']['metrics'][2],
                                    res['title'] + ': Farneback')

