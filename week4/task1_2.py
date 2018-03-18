from data import kitti
from video import optical_flow

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

        flow_lk = optical_flow.lk_sequence(seq, num_feat=100, q_feat=0.01,
                                           feat_dist=5, wsize=(15, 15))

        flow_lk_pyr = optical_flow.lk_pyr_sequence(seq, num_feat=100,
                                                   q_feat=0.01, feat_dist=5,
                                                   wsize=(15, 15), pyr_levels=3)

        flow_fnb = optical_flow.farneback_sequence(seq, levels=3, pyr_sc=0.5,
                                                   wsize=15, n_iter=10,
                                                   poly_n=7, p_sigma=1.5,
                                                   gauss=True)


