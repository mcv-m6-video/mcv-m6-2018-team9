import numpy as np

from data import cdnet
from video import bg_subtraction, morphology
from evaluation import metrics, animations


def run(dataset):
    train = cdnet.read_sequence('week3', dataset, 'train', colorspace='gray',
                                annotated=False)
    test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='gray', annotated=True)

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    # TODO: use parameters for each dataset
    pred = bg_subtraction.predict(test, model, 1.75, rho=0.10)

    # Apply imfill with 4- and 8-connectivity
    filled4 = morphology.imfill(pred, neighb=4)
    filled8 = morphology.imfill(pred, neighb=8)

    # TODO: join both batches in a unique gif image
    animations.video_recorder(pred, '', 'w3t1_orig')
    animations.video_recorder(filled4, '', 'w3t1_filled4')
    animations.video_recorder(filled8, '', 'w3t1_filled8')

    # Print F-scores
    summary_pred = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
    summary_fill4 = metrics.eval_from_mask(filled4, gt[:,0], gt[:,1])
    summary_fill8 = metrics.eval_from_mask(filled8, gt[:,0], gt[:,1])

    beta = 10
    fsco_pred = metrics.f_score(summary_pred, beta=beta)
    fsco_fill4 = metrics.f_score(summary_fill4, beta=beta)
    fsco_fill8 = metrics.f_score(summary_fill8, beta=beta)

    print('F-score (beta={}):'.format(beta))
    print('\t no fill: {}'.format(fsco_pred))
    print('\t fill4: {}'.format(fsco_fill4))
    print('\t fill8: {}'.format(fsco_fill8))
